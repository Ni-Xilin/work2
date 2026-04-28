import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, emb_size=64, input_size=1000):
        super().__init__()

        if input_size == 1000:
            emb_in = 1024
            self.last_pool = [8, 4, 3]
        elif input_size == 1600:
            emb_in = 1536
            self.last_pool = [8, 4, 3]
        elif input_size == 400:
            emb_in = 1024
            self.last_pool = [4, 2, 2]
        elif input_size == 600:
            emb_in = 1536
            self.last_pool = [4, 2, 2]
        else:
            raise ValueError(f"input_size: {input_size} is not supported")

        self.dropout = nn.Dropout(p=0.1)

        self.conv11 = nn.Conv1d(1, 32, 8, padding="same")
        self.conv12 = nn.Conv1d(32, 32, 8, padding="same")

        self.conv21 = nn.Conv1d(32, 64, 8, padding="same")
        self.conv22 = nn.Conv1d(64, 64, 8, padding="same")

        self.conv31 = nn.Conv1d(64, 128, 8, padding="same")
        self.conv32 = nn.Conv1d(128, 128, 8, padding="same")

        self.conv41 = nn.Conv1d(128, 256, 8, padding="same")
        self.conv42 = nn.Conv1d(256, 256, 8, padding="same")

        self.emb = nn.Linear(emb_in, emb_size)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)

        x = F.elu(self.conv11(x))
        x = F.elu(self.conv12(x))
        x = F.max_pool1d(x, 8, 4, padding=2)
        x = self.dropout(x)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool1d(x, 8, 4, padding=3)
        x = self.dropout(x)

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.max_pool1d(x, 8, 4, padding=3)
        x = self.dropout(x)

        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = F.max_pool1d(x, self.last_pool[0], self.last_pool[1], padding=self.last_pool[2])
        x = self.dropout(x)

        x = torch.reshape(x, (x.size(0), -1))
        x = torch.squeeze(self.emb(x))
        return x
