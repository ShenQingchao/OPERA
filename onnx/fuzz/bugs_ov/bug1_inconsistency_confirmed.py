import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/ConvTranspose.onnx'
session = ort.InferenceSession(onnx_model_path)


input_x = np.random.random([1, 1, 3, 3]).astype(np.float32)
input_w = np.random.random([1, 2, 3, 3]).astype(np.float32)
input_data = {"X": input_x, "W": input_w}

output_name = session.get_outputs()[0].name
onnx_output = session.run([output_name], input_data)[0]


ov_model = ov.convert_model(onnx_model_path)

ir_path = f"temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="CPU")

# show the model structure
# input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
# network_input_shape = input_key.shape

ov_result = compiled_model(input_data)[output_key]

np.testing.assert_allclose(onnx_output, ov_result, atol=1e-3)

# [confirmed]https://github.com/openvinotoolkit/openvino/issues/20848
