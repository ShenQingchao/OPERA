import torch
import openvino as ov
from torch.nn import Module


class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], kernel_size=2)


input_data = torch.randn([1, 5, 6, 7], dtype=torch.float32)
torch_model = avg_pool3d().float().eval()
trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)
input_shapes = input_data.shape
print(input_shapes)
torch_outputs = torch_model(input_data).cpu().numpy()
print(torch_outputs.shape)

ov_model = ov.convert_model(trace, example_input=input_data)
ir_path = f"_temp_model/temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="CPU")
output_key = compiled_model.output(0)
result = compiled_model(input_data)[output_key]

# [confirmed] https://github.com/openvinotoolkit/openvino/issues/20907
