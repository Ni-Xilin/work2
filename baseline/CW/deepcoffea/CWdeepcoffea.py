import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
import pathlib

# Import the DeepCoffea model definition
from target_model.Deepcoffea import Model as DeepCoffeaModel

# ------------------------------------------------------------
# 1. Data Handling Functions (from PGDdeepcoffea.py)
# ------------------------------------------------------------
def partition_single_session(session, delta, win_size, n_wins, tor_len, device):
    """
    Partitions a single session into windows, returning the windows, feature end indices, and original boundaries.
    """
    win_size_ms = win_size * 1000
    delta_ms = delta * 1000
    offset = win_size_ms - delta_ms

    session_ipd = session[0, :]
    session_size = session[1, :]
    cumulative_time = session_ipd.abs().cumsum(dim=0)
    
    partitioned_data_single = []
    time_indices_single = []
    size_indices_single = []
    original_boundaries_single = []

    for wi in range(int(n_wins)):
        start_time = wi * offset
        end_time = start_time + win_size_ms
        start_idx = torch.searchsorted(cumulative_time, start_time).item()
        end_idx = torch.searchsorted(cumulative_time, end_time).item()
        
        original_boundaries_single.append((start_idx, end_idx))

        window_ipd = session_ipd[start_idx:end_idx]
        window_size = session_size[start_idx:end_idx]

        if len(window_ipd) > 0:
            window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

        len_ipd = len(window_ipd)
        len_size = len(window_size)
        
        time_end_idx = len_ipd - 1 if len_ipd > 0 else -1
        size_end_idx = (len_ipd + len_size - 1) if len_size > 0 else -1
        
        final_tor_len = tor_len * 2
        window_data = torch.cat([window_ipd, window_size])
        if window_data.shape[0] < final_tor_len:
            padding = torch.zeros(final_tor_len - window_data.shape[0], device=device)
            window_data = torch.cat([window_data, padding])
        window_data = window_data[:final_tor_len]

        partitioned_data_single.append(window_data)
        time_indices_single.append(time_end_idx)
        size_indices_single.append(size_end_idx)
    
    partitioned_data = torch.stack(partitioned_data_single, dim=0)
    return partitioned_data, time_indices_single, size_indices_single, original_boundaries_single

def reconstruct_single_session(adv_windows_tensor, original_session, boundaries, time_indices, size_indices):
    """Reconstructs a full adversarial session flow from perturbed windows and original boundaries."""
    reconstructed_session = original_session.clone()
    n_wins = adv_windows_tensor.shape[0]

    for j in range(n_wins):
        adv_win = adv_windows_tensor[j]
        s_idx, e_idx = boundaries[j]
        
        if s_idx >= e_idx: continue
            
        t_end_in_win = time_indices[j]
        s_end_in_win = size_indices[j]

        len_ipd_in_win = t_end_in_win + 1
        len_size_in_win = s_end_in_win - t_end_in_win
        
        adv_win_ipd = adv_win[0 : len_ipd_in_win]
        adv_win_size = adv_win[len_ipd_in_win : len_ipd_in_win + len_size_in_win]
        
        if len(adv_win_ipd) > 1:
            reconstructed_session[0, s_idx + 1 : s_idx+len(adv_win_ipd[1:])+1] = adv_win_ipd[1:]
        if len(adv_win_size) > 0:
            reconstructed_session[1, s_idx : s_idx+len(adv_win_size)] = adv_win_size
            
    return reconstructed_session

# ------------------------------------------------------------
# 2. C&W L2 Attack for DeepCoffea
# ------------------------------------------------------------
def deepcoffea_cw_attack(anchor_model, pandn_model, original_tor_window, original_exit_window,
                         time_end_idx, size_end_idx, device,
                         binary_search_steps, num_iter_optim,
                         confidence, initial_c, lr, time_l2_weight):
    """
    Performs a C&W L2 attack on a DeepCoffea window pair to minimize cosine similarity.
    This version uses a weighted L2 loss for time and size features.
    """
    anchor_model.eval()
    pandn_model.eval()

    # Create a mask to only perturb valid (non-padding) data
    attack_mask = torch.zeros_like(original_tor_window).to(device)
    if time_end_idx >= 0:
        attack_mask[0:time_end_idx+1] = 1
    if size_end_idx > time_end_idx:
        attack_mask[time_end_idx+1:size_end_idx+1] = 1
    
    # Pre-calculate the embedding for the non-perturbed exit window
    with torch.no_grad():
        exit_embedding = pandn_model(original_exit_window.clone().detach().unsqueeze(0))

    best_similarity = float('inf')
    best_adv_window = original_tor_window.clone()

    lower_bound_c = 0.0
    upper_bound_c = initial_c

    final_time_ratio = 1.0
    final_size_ratio = 1.0
    flag = False
    for step in range(binary_search_steps):
        c = (lower_bound_c + upper_bound_c) / 2.0
        
        delta = torch.zeros_like(original_tor_window, requires_grad=True, device=device)
        optimizer = optim.Adam([delta], lr=lr)

        for i in range(num_iter_optim):
            optimizer.zero_grad()
            
            adv_window = original_tor_window + delta * attack_mask
            
            # Since DeepCoffea uses signed data, we don't clamp to min=0
            
            adv_embedding = anchor_model(adv_window.unsqueeze(0))
            
            # The "classification" loss for DeepCoffea is the cosine similarity.
            # We want to MINIMIZE it. The original C&W maximizes f(x_adv) for the wrong class.
            # Here, we can define the loss as simply the cosine similarity itself.
            loss_similarity = F.cosine_similarity(adv_embedding.unsqueeze(0), exit_embedding.unsqueeze(0)).mean()
            
            # Weighted L2 distance calculation
            original_tor_window = original_tor_window * attack_mask.to(device)
            perturbation = (adv_window - original_tor_window) * attack_mask
            pert_time = perturbation[0:time_end_idx+1] if time_end_idx >= 0 else torch.tensor([], device=device)
            pert_size = perturbation[time_end_idx+1:size_end_idx+1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
            
            l2_dist_time = torch.linalg.norm(pert_time, ord =2,dim = -1)
            l2_dist_size = torch.linalg.norm(pert_size, ord =2,dim = -1)
            
            origin_time_l2 = torch.linalg.norm(original_tor_window[0:time_end_idx+1], ord =2,dim = -1) if time_end_idx >= 0 else torch.tensor(1.0, device=device)
            origin_size_l2 = torch.linalg.norm(original_tor_window[time_end_idx+1:size_end_idx+1], ord =2,dim = -1) if size_end_idx > time_end_idx else torch.tensor(1.0, device=device)
            
            time_ratio = l2_dist_time / (origin_time_l2 + 1e-12)
            size_ratio = l2_dist_size / (origin_size_l2 + 1e-12)
            
            
            # The total loss is a combination of the similarity and weighted L2 norm of the perturbation
            loss_l2 = time_l2_weight * time_ratio + size_ratio
            total_loss = loss_l2 + c * (loss_similarity + confidence)
            
            total_loss.backward()
            optimizer.step()

        # Evaluate the attack for the current 'c'
        with torch.no_grad():
            final_adv_window = original_tor_window + delta * attack_mask
            final_similarity = F.cosine_similarity(anchor_model(final_adv_window.unsqueeze(0)).unsqueeze(0), exit_embedding.unsqueeze(0)).mean().item()
        

            original_tor_window = original_tor_window * attack_mask.to(device)
            perturbation = (adv_window - original_tor_window) * attack_mask
            pert_time = perturbation[0:time_end_idx+1] if time_end_idx >= 0 else torch.tensor([], device=device)
            pert_size = perturbation[time_end_idx+1:size_end_idx+1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
            
            l2_dist_time = torch.linalg.norm(pert_time, ord =2,dim = -1)
            l2_dist_size = torch.linalg.norm(pert_size, ord =2,dim = -1)
            
            origin_time_l2 = torch.linalg.norm(original_tor_window[0:time_end_idx+1], ord =2,dim = -1) 
            origin_size_l2 = torch.linalg.norm(original_tor_window[time_end_idx+1:size_end_idx+1], ord =2,dim = -1) 
            
            time_ratio = l2_dist_time / origin_time_l2 
            size_ratio = l2_dist_size / origin_size_l2 
            print(f"Step {step+1}/{binary_search_steps}, c: {c:.4f}, "
                  f"Time L2 Ratio: {time_ratio.item():.4f}, Size L2 Ratio: {size_ratio.item():.4f}, "
                  f"Output: {final_similarity:.6f}")
        # Update binary search bounds
        # If similarity is low enough (successful attack), try a smaller c to reduce perturbation
        if time_ratio.mean() > 0.15 or size_ratio.mean() > 0.15 : # Using 0 as a threshold for "dissimilar"
            upper_bound_c = c
            if not flag:
                if (l2_dist_time / origin_time_l2 ).mean()>0.15:
                    ratio = l2_dist_time / origin_time_l2
                    scaling_factor = torch.ones_like(ratio)
                    exceed = ratio > 0.15
                    scaling_factor[exceed] = 0.15 / ratio[exceed]
                    perturbation[0:time_end_idx+1] *= scaling_factor

                if ( l2_dist_size / origin_size_l2 ).mean()> 0.15:
                    ratio = l2_dist_size / origin_size_l2
                    scaling_factor = torch.ones_like(ratio)
                    exceed = ratio > 0.15
                    scaling_factor[exceed] = 0.15 / ratio[exceed]
                    perturbation[time_end_idx+1:size_end_idx+1] *= scaling_factor
                
                best_adv_window = original_tor_window + perturbation * attack_mask
        else: 
            if final_similarity < best_similarity:
                best_similarity = final_similarity
                best_adv_window = final_adv_window
                final_time_ratio = time_ratio
                final_size_ratio = size_ratio
            lower_bound_c = c
            flag = True


    try:    
        print(f"Best similarity achieved: {best_similarity:.6f}")
        print(f"Best perturbation ratios - Time: {final_time_ratio:.4f}, Size: {final_size_ratio:.4f}")
    except Exception as e:
        print(f"未攻击成功……")
    return best_adv_window.detach()


# ------------------------------------------------------------
# 3. Main Execution Function
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load DeepCoffea Model ---
    print("Loading pre-trained DeepCoffea model...")
    ckpt_path = pathlib.Path(args.target_model_path).resolve()
    fields = ckpt_path.name.split("_")
    emb_size = int(fields[-4].split("es")[-1])
    tor_len = int(fields[-8].split("tl")[-1])
    exit_len = int(fields[-7].split("el")[-1])

    anchor_model = DeepCoffeaModel(emb_size=emb_size, input_size=tor_len * 2).to(device)
    pandn_model = DeepCoffeaModel(emb_size=emb_size, input_size=exit_len * 2).to(device)

    model_checkpoint_path = os.path.join(args.target_model_path, 'best_loss.pth')
    state_dict = torch.load(model_checkpoint_path, map_location=device)
    anchor_model.load_state_dict(state_dict['anchor_state_dict'])
    pandn_model.load_state_dict(state_dict['pandn_state_dict'])
    anchor_model.eval()
    pandn_model.eval()
    print("Model loaded successfully.")

    # --- Load Dataset ---
    print("Loading dataset...")
    data_filename = f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}"
    session_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test_session.npz")
    win_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test.npz")

    test_data_session = np.load(session_path, allow_pickle=True)
    test_data_win = np.load(win_path)
    
    indices_to_attack = np.arange(args.num_samples)

    # --- Storage for results ---
    all_original_sims, all_attacked_sims = [], []
    time_l2_ratios, size_l2_ratios = [], []
    total_original_tor_windows = []
    total_original_exit_windows = []
    total_repartitioned_adv_windows = []
    print(f"Starting C&W attack on the first {args.num_samples} samples...")
    for idx in tqdm(indices_to_attack):
        tor_ipd = torch.tensor(test_data_session['tor_ipds'][idx], dtype=torch.float32).to(device)
        tor_size = torch.tensor(test_data_session['tor_sizes'][idx], dtype=torch.float32).to(device)
        original_session = torch.stack([tor_ipd, tor_size], dim=0)
        original_exit_windows = torch.tensor(test_data_win['test_exit'][:,idx,:], dtype=torch.float32).to(device)

        original_tor_windows, time_indices, size_indices, boundaries = partition_single_session(
            original_session, args.delta, args.win_size, args.n_wins, args.tor_len, device)
        
        # Attack each window individually
        adv_tor_windows_list = []
        for i in range(args.n_wins):
            print(f"Attacking window {i+1}/{args.n_wins} for sample {idx+1}/{args.num_samples}...")
            tor_win = original_tor_windows[i]
            exit_win = original_exit_windows[i]
            t_end_idx = time_indices[i]
            s_end_idx = size_indices[i]

            # Skip empty windows
            if t_end_idx < 0 and s_end_idx < 0:
                adv_tor_windows_list.append(tor_win)
                continue

            adv_win = deepcoffea_cw_attack(
                anchor_model, pandn_model, tor_win, exit_win,
                t_end_idx, s_end_idx, device,
                args.binary_search_steps, args.num_iter,
                args.confidence, args.initial_c, args.lr,
                args.time_l2_weight
            )
            adv_tor_windows_list.append(adv_win)
        
        adv_tor_windows = torch.stack(adv_tor_windows_list, dim=0)
        
        # Reconstruct and re-partition the adversarial session
        reverted_session = reconstruct_single_session(
                    adv_tor_windows, original_session, 
                    boundaries, time_indices, size_indices
            )
        repartitioned_adv_windows, _, _, _ = partition_single_session(
                reverted_session, args.delta, args.win_size, args.n_wins, args.tor_len, device
            )

        # Evaluate similarity and perturbation
        with torch.no_grad():
            orig_tor_emb = anchor_model(original_tor_windows)
            orig_exit_emb = pandn_model(original_exit_windows)
            all_original_sims.append(F.cosine_similarity(orig_tor_emb, orig_exit_emb).mean().item())

            adv_tor_emb = anchor_model(repartitioned_adv_windows)
            all_attacked_sims.append(F.cosine_similarity(adv_tor_emb, orig_exit_emb).mean().item())

            # Calculate perturbation ratios based on the initial window-wise perturbations
            total_perturbation = adv_tor_windows - original_tor_windows
            session_time_ratios, session_size_ratios = [], []
            for i in range(args.n_wins):
                t_end_idx, s_end_idx = time_indices[i], size_indices[i]
                pert_time = total_perturbation[i, 0:t_end_idx+1] if t_end_idx >= 0 else torch.tensor([], device=device)
                orig_time = original_tor_windows[i, 0:t_end_idx+1] if t_end_idx >= 0 else torch.tensor([], device=device)
                pert_size = total_perturbation[i, t_end_idx+1:s_end_idx+1] if s_end_idx > t_end_idx else torch.tensor([], device=device)
                orig_size = original_tor_windows[i, t_end_idx+1:s_end_idx+1] if s_end_idx > t_end_idx else torch.tensor([], device=device)
                
                time_ratio_val = (torch.linalg.norm(pert_time, ord =2,dim = -1) / (torch.linalg.norm(orig_time, ord =2,dim = -1) + 1e-12)).mean().item()
                size_ratio_val = (torch.linalg.norm(pert_size, ord =2,dim = -1) / (torch.linalg.norm(orig_size, ord =2,dim = -1) + 1e-12)).mean().item()
                session_time_ratios.append(time_ratio_val)
                session_size_ratios.append(size_ratio_val)
            
            time_l2_ratios.append(np.nanmean(session_time_ratios))
            size_l2_ratios.append(np.nanmean(session_size_ratios))
            total_repartitioned_adv_windows.append(repartitioned_adv_windows.cpu())
            total_original_tor_windows.append(original_tor_windows.cpu())
            total_original_exit_windows.append(original_exit_windows.cpu())

    # --- Print Final Results ---
    print("\n--- C&W Attack on DeepCoffea: Final Results ---")
    print(f"Attacked Samples: {args.num_samples}")
    print(f"Average Similarity BEFORE Attack: {np.mean(all_original_sims):.4f}")
    print(f"Average Similarity AFTER Attack:  {np.mean(all_attacked_sims):.4f}")
    print("-" * 20)
    print(f"Average L2 Perturbation Ratio (Time): {np.mean(time_l2_ratios):.4f}")
    print(f"Average L2 Perturbation Ratio (Size): {np.mean(size_l2_ratios):.4f}")
    
    # --- Save results to a file ---
    result_fpath = pathlib.Path(f'baseline/CWdeepcoffea_advsamples_time{np.mean(time_l2_ratios):.4f}_size{np.mean(size_l2_ratios):.4f}.p')
    result_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(result_fpath, "wb") as fp:
        results = {
            "adv_tor_windows": total_repartitioned_adv_windows,
            "original_tor_windows": total_original_tor_windows,
            "original_exit_windows": total_original_exit_windows,
            "original_sims": np.mean(all_original_sims),
            "attacked_sims": np.mean(all_attacked_sims),
            "avg_time_l2_ratio": np.mean(time_l2_ratios),
            "avg_size_l2_ratio": np.mean(size_l2_ratios),
        }
        pickle.dump(results, fp)
    print(f"Results saved to {result_fpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="C&W L2 attack on DeepCoffea")
    # Paths and Data Parameters
    parser.add_argument("--target_model_path", type=str, default="target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/", help="Path to DeepCoffea model folder.")
    parser.add_argument("--data_path", type=str, default="target_model/deepcoffea/dataset/CrawlE_Proc/", help="Path to dataset root directory.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Computation device.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to attack.")

    # Data Parameters (should match model and data filenames)
    parser.add_argument("--delta", default=3, type=int)
    parser.add_argument("--win_size", default=5, type=int)
    parser.add_argument("--n_wins", default=11, type=int)
    parser.add_argument("--threshold", default=20, type=int)
    parser.add_argument("--tor_len", default=500, type=int)
    parser.add_argument("--exit_len", default=800, type=int)
    parser.add_argument("--n_test", default=1000, type=int)

    # C&W Attack Hyperparameters
    parser.add_argument("--confidence", type=float, default=0, help="Confidence parameter (kappa) for C&W loss. Higher means stronger attack.")
    parser.add_argument("--initial_c", type=float, default=20, help="Initial value for the constant 'c' in the binary search.")
    parser.add_argument("--lr", type=float, default=0.4, help="Learning rate for the Adam optimizer in the attack.")
    parser.add_argument("--binary_search_steps", type=int, default=15, help="Number of steps for the binary search over 'c'.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of optimization iterations for each binary search step.")
    # parser.add_argument("--time_l2_weight", type=float, default=0.1, help="Weight for the L2 loss of time features. Use a value < 1.0 if time perturbations are too large.")
    parser.add_argument("--time_l2_weight", type=float, default=0.5, help="Weight for the L2 loss of time features. Use a value < 1.0 if time perturbations are too large.")
    
    args = parser.parse_args()
    main(args)