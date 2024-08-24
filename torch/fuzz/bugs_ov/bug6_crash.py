import torch
import openvino as ov

torch_model = torch.nn.Hardtanh(0.0, 6.0, False,).eval()

input_data = torch.randint(1, 100, [0, 2], dtype=torch.int8)  # only 0 can trigger this bug. this may be a feature
print(input_data)
trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)
input_shapes = input_data.shape
print(input_shapes)
torch_outputs = torch_model(input_data).cpu().numpy()
print(torch_outputs)

ov_model = ov.convert_model(trace, example_input=input_data)
ir_path = f"_temp_model/temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="CPU")
output_key = compiled_model.output(0)
result = compiled_model(input_data)[output_key]
