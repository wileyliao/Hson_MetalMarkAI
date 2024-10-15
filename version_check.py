import torch
print(torch.backends.cudnn.version())

def check_pytorch():
    print("\nPyTorch Check:")
    # 1. 框架的版本
    torch_version = torch.__version__
    print(f"  Installed PyTorch version: {torch_version}")

    # 2. 框架是否支援該 CUDA & CUDA 版本
    cuda_available_pt = torch.cuda.is_available()
    if cuda_available_pt:
        cuda_version_pt = torch.version.cuda
        print(f"  PyTorch CUDA Support: {cuda_available_pt}, CUDA Version: {cuda_version_pt}")
    else:
        print("  PyTorch CUDA Support: Not available")

    # 3. 框架是否支援該 cuDNN & cuDNN 版本
    cudnn_available_pt = torch.backends.cudnn.is_available()
    if cudnn_available_pt:
        cudnn_version_pt = torch.backends.cudnn.version()
        print(f"  PyTorch cuDNN Support: {cudnn_available_pt}, cuDNN Version: {cudnn_version_pt}")
    else:
        print("  PyTorch cuDNN Support: Not available")

    # 4. 列出可用的 GPU
    if cuda_available_pt:
        print("  Available GPU Devices:")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  No GPU devices available with PyTorch")

# 執行檢查
check_pytorch()