import torch
import paddle
import sys

print(f"Python version: {sys.version}")
print("-" * 30)

# Check Torch GPU
torch_cuda = torch.cuda.is_available()
print(f"Torch GPU available: {torch_cuda}")
if torch_cuda:
    print(f"Torch GPU device: {torch.cuda.get_device_name(0)}")

print("-" * 30)

# Check Paddle GPU
paddle_cuda = paddle.is_compiled_with_cuda()
print(f"Paddle GPU available: {paddle_cuda}")
if paddle_cuda:
    # Get device list
    device_name = paddle.get_device()
    print(f"Paddle current device: {device_name}")
    
print("-" * 30)
