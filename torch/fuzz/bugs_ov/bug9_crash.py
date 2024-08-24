import torch
from torch.nn import Module
import openvino as ov


input_data = torch.randn([2, 5], dtype=torch.float32)
class max_pool1d(Module):
    def forward(self, *args):
        torch.nn.MaxPool1d
        return torch.nn.functional.max_pool1d(args[0], kernel_size=3,)

torch_model = max_pool1d().float().eval()
torch_outputs = torch_model(input_data).cpu().numpy()
print(torch_outputs)

trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)

input_shapes = input_data.shape
ov_model = ov.convert_model(trace, example_input=input_data)

# [] https://github.com/openvinotoolkit/openvino/issues/21052
