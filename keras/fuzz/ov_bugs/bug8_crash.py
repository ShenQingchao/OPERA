import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np
import openvino as ov

layer = keras.layers.BatchNormalization()
input_shape = [1, 2, 3, 4, 5]
input_data = np.random.random(input_shape).astype(np.float32)

x = layers.Input(shape=input_shape[1:], dtype="float32")
y = layer(x)
model = models.Model(x, y)
# model.summary()
res_keras = model(input_data)

tf2_model_path = f"_temp_model"
tf.saved_model.save(model, tf2_model_path)
ov_model = ov.convert_model(tf2_model_path,  input=input_shape)

# [] https://github.com/openvinotoolkit/openvino/issues/21247
