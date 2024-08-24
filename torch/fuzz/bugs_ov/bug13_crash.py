import torch
from torch.nn import Module
import openvino as ov


class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], kernel_size=2)


input_data = torch.randint(1, 10, [1, 2, 3], dtype=torch.int64)
torch_model = avg_pool1d().float().eval()
torch_outputs = torch_model(input_data).cpu().numpy()

trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)

input_shapes = input_data.shape
ov_model = ov.convert_model(trace, example_input=input_data)

# [] https://github.com/openvinotoolkit/openvino/issues/21071
