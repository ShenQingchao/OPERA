import openvino as ov
import numpy as np

ov_model = ov.convert_model('../_temp_model/dropout.onnx')  # input=input_shapes


ir_path = f"temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="CPU")


input_x = np.random.random([3,  4, 5]).astype(np.float32)
input_data = {"x": input_x}

result = []
for output in ov_model.outputs:
    result.append(compiled_model(input_data)[output])
