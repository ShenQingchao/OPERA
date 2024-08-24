import openvino as ov

ov_model = ov.convert_model('../_temp_model/0.onnx')  # input=input_shapes
