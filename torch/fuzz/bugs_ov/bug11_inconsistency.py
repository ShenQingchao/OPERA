import torch
from torch.nn import Module
import openvino as ov
import numpy as np


def compile_torch(model, input_data):
    ov_model = ov.convert_model(model, example_input=input_data)
    ir_path = f"_temp_model/temp_OVIR.xml"
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    core = ov.Core()
    model = core.read_model(ir_path)

    compiled_model = core.compile_model(model=model, device_name="CPU")  # crash here
    output_key = compiled_model.output(0)
    result = compiled_model(input_data)[output_key]
    return result


class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], kernel_size=3, dilation=1, ceil_mode=True, )


input_data = torch.randn([5, 174, 2, 4], dtype=torch.float32)
torch_model = max_pool2d().float().eval()
torch_outputs = torch_model(input_data).cpu().numpy()

trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)

input_shapes = input_data.shape
res_ov = compile_torch(trace, input_data)

# [] https://github.com/openvinotoolkit/openvino/issues/21067
