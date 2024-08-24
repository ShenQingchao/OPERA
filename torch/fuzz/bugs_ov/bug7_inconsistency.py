import torch
from torch.nn import Module
import openvino as ov
import numpy as np


def compile_torch(model, input_data):
    ov_model = ov.convert_model(model, example_input=input_data)
    ir_path = f"_temp_model/temp_OVIR.xml"
    ov.save_model(ov_model, ir_path)
    core = ov.Core()
    model = core.read_model(ir_path)

    compiled_model = core.compile_model(model=model, device_name='CPU')
    output_key = compiled_model.output(0)
    result = compiled_model(input_data)[output_key]
    return result

input_data = torch.randn([1, 2, 3], dtype=torch.float32)


class lp_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.lp_pool1d(args[0], norm_type=1.5, kernel_size=1)

torch_model = lp_pool1d().float().eval()
torch_outputs = torch_model(input_data).cpu().numpy()

trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)

input_shapes = input_data.shape
res_ov = compile_torch(trace, input_data)
np.testing.assert_allclose(torch_outputs, res_ov, rtol=1e-3, atol=1e-3)

# [] https://github.com/openvinotoolkit/openvino/issues/21044

