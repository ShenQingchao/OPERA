import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/Trilu.onnx'
# session = ort.InferenceSession(onnx_model_path)
#
# input_x = np.random.randint(0, 10, [4, 5], dtype=np.int64)
# k = np.random.randint(0, 1, size=[1], dtype=np.int64)
# input_data = {"x": input_x, "k": k,}
#
# output_name = session.get_outputs()[0].name
# onnx_output = session.run(None, input_data)

ov_model = ov.convert_model(onnx_model_path)

# [] https://github.com/openvinotoolkit/openvino/issues/21172
