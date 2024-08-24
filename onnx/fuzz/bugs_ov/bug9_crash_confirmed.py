import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/0.onnx'
session = ort.InferenceSession(onnx_model_path)
x = np.random.random([10]).astype(np.float64)
input_data = {"x": x}

output_name = session.get_outputs()[0].name
onnx_output = session.run(None, input_data)
print(onnx_output)
ov_model = ov.convert_model(onnx_model_path)

# [confirmed] https://github.com/openvinotoolkit/openvino/issues/21173
