import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np

import openvino as ov

layer = keras.layers.GlobalAveragePooling2D()
input_shape = [1, 11, 1, 7]
input_data = np.random.randint(2, size=input_shape)
x = layers.Input(shape=input_shape[1:], dtype="int8")
y = layer(x)
model = models.Model(x, y)
# model.summary()
res_keras = model(input_data)

tf2_model_path = f"_temp_model"
tf.saved_model.save(model, tf2_model_path)
ov_model = ov.convert_model(tf2_model_path,  input=input_shape)

ir_path = f"_temp_OVIR.xml"
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)
compiled_model = core.compile_model(model=model, device_name="CPU")  # GPU: run well

output_key = compiled_model.output(0)

res_dlc = compiled_model(input_data)[output_key]
print(res_keras)
print(res_dlc)
np.testing.assert_allclose(res_keras, res_dlc, atol=1e-3, rtol=1e-3)
# https://github.com/openvinotoolkit/openvino/issues/20815
