import torch
from torch.nn import Module
import openvino as ov


class normalize(Module):
    def forward(self, *args):
        return torch.nn.functional.normalize(args[0], eps=2)


input_data = torch.randn([15, 15, 17], dtype=torch.float32)
torch_model = normalize().float().eval()
torch_outputs = torch_model(input_data).cpu().numpy()

trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)

input_shapes = input_data.shape
ov_model = ov.convert_model(trace, example_input=input_data)

# [https://github.com/openvinotoolkit/openvino/issues/21066]
