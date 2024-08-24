import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np

import openvino as ov

layer = keras.layers.Permute(dims=[])
input_shape = [2]
input_data = np.asarray([0.1, 0.2])

print(input_data)
x = layers.Input(shape=input_shape[1:], dtype="float32")
y = layer(x)
model = models.Model(x, y)
model.summary()
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
np.testing.assert_allclose(res_keras, res_dlc, atol=1e-3, rtol=1e-3)

# [confirmed] https://github.com/openvinotoolkit/openvino/issues/20821
