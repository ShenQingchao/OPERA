# [] https://github.com/openvinotoolkit/openvino/issues/20822
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np
import openvino as ov

layer = keras.layers.Attention(causal=True)
input_shape = [1, 3, 1]
input1 = np.random.random(input_shape)
input_data = [input1, input1]
weights = layer.get_weights()
layer.set_weights(weights)

x1 = layers.Input(shape=input_shape[1:], dtype="float32")
x2 = layers.Input(shape=input_shape[1:], dtype="float32")
x = [x1, x2]
y = layer(x)
model = models.Model(x, y)
model.summary()
res_keras = model(input_data)
tf.saved_model.save(model, "tf_model")
ov_model = ov.convert_model("tf_model",  input=input_shape)

# [fixed] https://github.com/openvinotoolkit/openvino/issues/20822
