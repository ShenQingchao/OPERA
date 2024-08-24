import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np
import openvino as ov

layer = keras.layers.UpSampling3D()
input_shape = [1, 2, 3, 4, 5]
input_data = np.random.randint(0, 1, size=input_shape)
weights = layer.get_weights()
layer.set_weights(weights)

x = layers.Input(shape=input_shape[1:], dtype="int16")
y = layer(x)
model = models.Model(x, y)
model.summary()
res_keras = model(input_data)
tf.saved_model.save(model, "tf_model")
ov_model = ov.convert_model("tf_model",  input=input_shape)

# [crash] Input element type must be f32, f16, bf16, i8, u8, i64, i32
# layer_test(keras.layers.UpSampling3D,args=(),kwargs={},input_shape=[20, 11, 3, 19, 2],input_dtype='int16',)
