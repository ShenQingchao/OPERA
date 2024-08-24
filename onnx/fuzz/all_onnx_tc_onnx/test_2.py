import onnx
from test_openvino_onnx import make_graph

make_graph(op_type='MaxPool', kwargs={'auto_pad': b'SAME_UPPER', 'kernel_shape': [3, 3], 'strides': [2, 2]},input_name=('x',), input_shape=([1, 1, 5, 5],), input_dtype=('FLOAT',), output_name=('y',),output_shape=([1, 1, 3, 3],), output_dtype=('FLOAT',),count=2,)