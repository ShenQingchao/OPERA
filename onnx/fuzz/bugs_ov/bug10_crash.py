import onnxruntime as ort
import openvino as ov
import numpy as np

onnx_model_path = '../_temp_model/RandomUniformLike.onnx'
ov_model = ov.convert_model(onnx_model_path)

# [confirmed]
