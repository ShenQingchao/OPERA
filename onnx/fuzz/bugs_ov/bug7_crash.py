import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/ReduceL1.onnx'
session = ort.InferenceSession(onnx_model_path)

data = np.random.random([3, 2, 2]).astype(np.float32)
axes = np.random.randint(0, 2, size=[1], dtype=np.int64)

input_data = {"data": data, "axes": axes}

output_name = session.get_outputs()[0].name
onnx_output = session.run(None, input_data)


ov_model = ov.convert_model(onnx_model_path)

ir_path = f"temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="CPU")
output_key = compiled_model.outputs


for i, output in enumerate(output_key):
    ov_output = compiled_model(input_data)[output]

