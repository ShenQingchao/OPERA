import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/Range.onnx'

ov_model = ov.convert_model(onnx_model_path)

# [confirmed] https://github.com/openvinotoolkit/openvino/issues/21178
