import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np

import openvino as ov

np.random.seed(1)
input_shape1 = [1, 2, 1]
input_shape2 = [1, 2, 2]
input_data1 = np.random.random(input_shape1).astype(np.float32)
input_data2 = np.random.random(input_shape2).astype(np.float32)
print(input_data1)
print(input_data2)


def test_concat():
    layer = keras.layers.Concatenate(axis=2)

    x1 = layers.Input(shape=input_shape1[1:], dtype="float32")
    x2 = layers.Input(shape=input_shape2[1:], dtype="float32")
    y = layer([x1, x2])
    model = models.Model([x1, x2], y)
    # model.summary()
    res_keras = model([input_data1, input_data2])
    tf2_model_path = f"_temp_model"
    tf.saved_model.save(model, tf2_model_path)
    ov_model = ov.convert_model(tf2_model_path,  input=[input_shape1, input_shape2])

    ir_path = f"_temp_OVIR.xml"
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    core = ov.Core()
    model = core.read_model(ir_path)
    compiled_model = core.compile_model(model=model, device_name="GPU")  # GPU: run well

    output_key = compiled_model.output(0)

    res_dlc = compiled_model([input_data1, input_data2])[output_key]
    print(res_dlc)
    np.testing.assert_allclose(res_keras, res_dlc, atol=1e-3, rtol=1e-3)


for i in range(5):
    test_concat()
    print(f"passed for item: {i}")
# []
