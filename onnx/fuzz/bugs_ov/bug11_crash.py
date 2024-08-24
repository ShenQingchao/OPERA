import openvino as ov

onnx_model_path = '../_temp_model/Split.onnx'
ov_model = ov.convert_model(onnx_model_path)

