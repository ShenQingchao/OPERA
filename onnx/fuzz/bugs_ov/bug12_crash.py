import openvino as ov
import numpy as np
import onnxruntime as ort
onnx_model_path = '../_temp_model/Pads.onnx'
session = ort.InferenceSession(onnx_model_path)

x = np.random.random([1, 3, 4, 5]).astype(np.float32)
pads = np.random.randint(0, 2, size=[8], dtype=np.int64)
input_data = {"x": x, "pads": pads}
output_name = session.get_outputs()[0].name
onnx_output = session.run(None, input_data)

ov_model = ov.convert_model(onnx_model_path)  # crash
