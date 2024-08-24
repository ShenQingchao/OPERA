# [Confirmed] [bugs x3] https://github.com/openvinotoolkit/openvino/issues/20811
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np

import openvino as ov

layer = keras.layers.ReLU()
input_shape = [1, 2, 3, 4]
input_data = np.random.randint(2, size=input_shape)
# weights = layer.get_weights()
# layer.set_weights(weights)

x = layers.Input(shape=input_shape[1:], dtype="uint64")  # uint16 and uint32 also trigger
y = layer(x)
model = models.Model(x, y)
# model.summary()
res_keras = model(input_data)

tf2_model_path = f"_temp_model"
tf.saved_model.save(model, tf2_model_path)
ov_model = ov.convert_model(tf2_model_path,  input=input_shape)
