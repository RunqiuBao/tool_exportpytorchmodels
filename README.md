# Register custom layers into pytorch, as well as tensorrt
- **ModulatedDeformConv2d**:
```
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
import ctypes; import torch; import os
# Note: build this repo first before this
torch.ops.load_library("/root/code/tool_exportpytorchmodels/dist/lib/libmmdeploy_torch_ops.so")
ctypes.CDLL(os.path.join("/root/code/tool_exportpytorchmodels/dist/lib/libmmdeploy_trt_ops.so"))
if hasattr(torch.ops, 'mmdeploy'):
    print("Success: Registered Ops:", [op for op in dir(torch.ops.mmdeploy) if not op.startswith('_')])
# Define a replacement forward function that uses the registered op
def patched_forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor):
    # Handle the case where bias is None (common in some backbones)
    bias_tensor = self.bias
    if bias_tensor is None:
        # Create a dummy empty tensor. 
        # The C++ kernel won't read it because we pass with_bias=False,
        # but the Schema Parser demands a valid Tensor object here.
        bias_tensor = torch.empty(0, device=x.device, dtype=x.dtype)
    # This matches the schema you defined in bind.cpp for modulated_deform_conv
    return torch.ops.mmdeploy.modulated_deform_conv(
        x, 
        self.weight, 
        bias_tensor,   # <--- NOW ALWAYS A TENSOR (Never None)
        offset, 
        mask,
        self.kernel_size[0], self.kernel_size[1],
        self.stride[0], self.stride[1],
        self.padding[0], self.padding[1],
        self.dilation[0], self.dilation[1],
        self.groups, 
        self.deform_groups, 
        self.bias is not None # Flag tells C++ whether to use the tensor
    )
@torch.library.register_fake("mmdeploy::modulated_deform_conv")
def _modulated_deform_conv_fake(input, weight, bias, offset, mask, 
                                kernel_h, kernel_w, stride_h, stride_w, 
                                pad_h, pad_w, dilation_h, dilation_w, 
                                groups, deform_groups, with_bias):
    
    # 1. Get input dimensions
    batch_size, _, in_h, in_w = input.shape
    out_channels = weight.shape[0] # Weight is (C_out, C_in/g, kH, kW)

    # 2. Calculate Output Height
    # Formula: floor((H + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    
    # 3. Calculate Output Width
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    # 4. Return an empty tensor on the 'meta' device with the correct shape and dtype
    return input.new_empty((batch_size, out_channels, out_h, out_w))
ModulatedDeformConv2d.forward = patched_forward
```

