import onnx
from test_openvino_onnx import make_graph

make_graph(op_type='Expand', kwargs={}, input_name=('X', 'shape'), input_shape=([1, 3, 1], [2]), input_dtype=('FLOAT', 'INT64'), output_name=('Y',), output_shape=([1, 3, 1],), output_dtype=('FLOAT',),count=1,)