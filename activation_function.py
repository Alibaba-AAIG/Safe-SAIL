import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from abc import ABC, abstractmethod

# 定义一个抽象激活函数类
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


# 实现 ReLU 激活函数
class ReLU(ActivationFunction):
    def forward(self, x):
        return F.relu(x)


# 实现 TopKReLU 激活函数
class TopKReLU(ActivationFunction):
    def __init__(self, k=1000):
        self.k = k

    def forward(self, x):
        k_values, _ = torch.topk(x, k=self.k, sorted=False)
        x_threshold = k_values.min(dim=-1, keepdim=True)[0]
        output = torch.where(x < x_threshold, torch.tensor(0.0, device=x.device), x)
        output = F.relu(output)
        return output

class RectangleFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # Convert the input to a tensor
        output = ((x > -0.5) & (x < 0.5)).to(x.dtype)
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = torch.zeros_like(x)  # gradient w.r.t. input is zero
        return grad_input

class JumpReLUFunction(Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        out = x * (x > threshold).to(x.dtype)
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth  # Save bandwidth for backward pass
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        # Gradient with respect to x is always zero for the step function
        x_grad = (x > threshold).to(x.dtype) * grad_output

        # Gradient with respect to the threshold
        rectangle = RectangleFunction.apply
        threshold_grad = (
            - (threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
        )
        
        return x_grad, threshold_grad, None  # No gradient for bandwidth


# 实现 JumpReLU 激活函数，以及手写backward
class JumpReLU(ActivationFunction):
    def __init__(self):
        self.bandwidth = 0.001
        self.jumprelu_function = JumpReLUFunction.apply
    
    def forward(self, x, theta):
        out = self.jumprelu_function(x, theta, self.bandwidth)
        return out

    def __call__(self, x, theta):
        return self.forward(x, theta)