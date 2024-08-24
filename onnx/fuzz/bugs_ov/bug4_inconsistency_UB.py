import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/Mod.onnx'
session = ort.InferenceSession(onnx_model_path)


input_x = np.array([0, 0, 0, 1, 1, 0], dtype=np.int64)
input_y = np.array([1, 1, 0, 1, 0, 1], dtype=np.int64)
input_data = {"x": input_x, "y":input_y}

output_name = session.get_outputs()[0].name
onnx_output = session.run(None, input_data)


ov_model = ov.convert_model(onnx_model_path)

ir_path = f"temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="GPU")

# show the model structure
# input_key = compiled_model.input(0)
output_key = compiled_model.outputs
# network_input_shape = input_key.shape

for i, output in enumerate(output_key):
    ov_output = compiled_model(input_data)[output]
    np.testing.assert_allclose(onnx_output[i], ov_output, atol=1e-3)

# [UB] https://github.com/openvinotoolkit/openvino/issues/20898
