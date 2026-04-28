"""通用训练入口。

这个脚本与 run_train_stub.py 使用同一套实现，
只是名称更贴近当前已经同时支持 stub 与 torch_real 的状态。
"""

from run_train_stub import main


if __name__ == "__main__":
    main()
