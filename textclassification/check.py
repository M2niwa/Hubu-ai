import torch
print("torch.cuda.is_available()")  # 应输出 True
print(torch.version.cuda)         # 应输出 CUDA 版本号
print(torch.cuda.get_device_name(0))  # 输出 GPU 名称
print(torch.__version__)          # 检查PyTorch版本
print(torch.cuda.is_available())  # 应为True
