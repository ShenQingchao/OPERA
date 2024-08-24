import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import traceback
import re
import openvino as ov
import logging
import os

logging.basicConfig(level=logging.ERROR)
np.random.seed(2023)


def extract_crash_message(e):
    print(e)
    tb = traceback.extract_tb(e.__traceback__)
    file_name, line_number, _, _ = tb[-1]
    file_name = file_name.split("site-packages")[-1]
    exc_type = type(e).__name__
    # stack_trace = str(e).strip().split("\n")[-1]
    stack_trace = str(e).split("Summary:")[0].strip().split("\n")[-1]
    # if stack_trace.endswith(':'):
    #     stack_trace = stack_trace[:-1]
    # stack_trace = stack_trace.split(':')[-1].strip()
    pattern = r"[\[\(].*?[\]\)]"
    stack_trace = re.sub(pattern, "", stack_trace)
    print(f">>>>>>>>>>>>>>>>>>>Bug Info: {stack_trace}")
    crash_message = f"{exc_type}_{file_name}_{line_number}_{stack_trace}"
    return crash_message


def record_bug(bug_id, bug_type, op, crash_message=''):
    bug_info_str = f"{bug_id}\t{bug_type}\t{op}\t{crash_message}\n"
    with open("../data/detected_bugs_ov.txt", 'a', encoding='utf-8') as f:
        f.write(bug_info_str)


def parse_input_shape(input_shape, max_dim_value=32):
    """
    1. assign a value for None
    2. if dim_value > 32, dim_value = 32, for fear OOM.
    :param input_shape:
    :return:
    """
    new_shape = []
    input_shape = list(input_shape)
    for i, elem in enumerate(input_shape):
        if isinstance(elem, list) or isinstance(elem, tuple):
            sub_shape = parse_input_shape(elem)
            new_shape.append(sub_shape)
        elif elem is None:
            if i == 0:  # batch_size for multiple input must be same
                new_shape.append(1)
            else:
                new_shape.append(np.random.randint(1, max_dim_value))
        elif isinstance(elem, int):
            new_shape.append(min(elem, max_dim_value))
        else:
            raise TypeError(f"unsolved Type for {input_shape}")
    return new_shape


def assign_input_data(input_shape, input_dtype):
    # assign value to input_data
    if isinstance(input_shape[0], int):
        input_data = 10 * np.random.random(input_shape)
        if input_dtype[:5] == "float":
            input_data -= 0.5
            input_data = input_data.astype(input_dtype)
        elif input_dtype[:3] == 'int':
            input_data = np.random.randint(2, size=input_shape)
    elif isinstance(input_shape[0], list):
        input_data = []
        for sub_shape in input_shape:
            sub_input_data = assign_input_data(sub_shape, input_dtype)
            input_data.append(sub_input_data)
    else:
        raise TypeError(f"Unresolved Type {type(input_shape[0])} for input_shape: {input_shape}")
    return input_data


def correct_shape(input_shape, layer_name, max_dim_value=32):
    if input_shape is None:
        if '3D' in layer_name:
            return [np.random.randint(1, max_dim_value) for i in range(5)]
        elif '1D' in layer_name:
            return [np.random.randint(1, max_dim_value) for i in range(3)]
        return [np.random.randint(1, max_dim_value) for i in range(4)]  # default input_shape in 4 dim

    if '1D' in layer_name:
        if len(input_shape) == 4:
            input_shape = input_shape[1:]
        elif len(input_shape) == 2:
            input_shape.insert(0, 1)
    if '2D' in layer_name:
        if len(input_shape) == 5:
            input_shape = input_shape[1:]
        elif len(input_shape) == 3:
            input_shape.insert(0, 1)
    elif '3D' in layer_name:
        if len(input_shape) == 6:
            input_shape = input_shape[1:]
        elif len(input_shape) == 4:
            input_shape = input_shape.insert(0, 1)
    return input_shape


def layer_test(
        layer_cls,
        args=None,
        kwargs=None,
        input_shape=None,
        input_dtype="float32",
        count=0,
        **unused_kwargs,
):
    print(count)
    try:
        kwargs = kwargs or {}
        args = args or ()

        input_shape = correct_shape(input_shape, layer_cls.__name__)
        input_shape = parse_input_shape(input_shape)
        input_data = assign_input_data(input_shape, input_dtype)  # Notice, input_shape will be changed.

        layer = layer_cls(*args, **kwargs)
        weights = layer.get_weights()
        layer.set_weights(weights)

        # test in functional API
        if isinstance(input_shape[0], list):  # multiple inputs
            x = []
            for sub_shape in input_shape:
                input_i = layers.Input(shape=sub_shape[1:], dtype=input_dtype)
                x.append(input_i)
        else:
            x = layers.Input(shape=input_shape[1:], dtype=input_dtype)
        y = layer(x)

        # check shape inference
        model = models.Model(x, y)
        # model.save("")
        # res_keras = model.predict(input_data)
        # model.summary()
        res_keras = model(input_data)
        # print('debug,', input_shape, input_data.shape)
        temp_model_dir = "_temp_model"
        if not os.path.exists(temp_model_dir):
            os.mkdir(temp_model_dir)
        tf2_model_path = f"{temp_model_dir}/_temp_model_{count}"
        tf.saved_model.save(model, tf2_model_path)
    except Exception as e:
        print("[keras error]", e)
        return
    try:
        res_dlc = compile_keras(count, tf2_model_path, input_shape, input_data, temp_model_dir)
    except Exception as e:
        if 'support' in str(e) or 'not allowed' in str(e) or "No conversion rule" in str(e):
            print(e)
            print("[Warning] trigger an unsupported behavior")
        else:
            print(f'[Bug in DLC] using test: {layer_cls}; id= {count}')
            print(e)
            crash_message = extract_crash_message(e)
            record_bug(count, 'crash', type(layer).__name__, crash_message=crash_message)
        return
    try:
        if isinstance(res_keras, list):  # multiple output:
            for i in range(len(res_keras)):
                np.testing.assert_allclose(res_keras[i], res_dlc[i], atol=1e-3, rtol=1e-3)
        else:
            np.testing.assert_allclose(res_keras, res_dlc[0], atol=1e-3, rtol=1e-3)
    except AssertionError as e:
        print(f'[Bug in DLC] using test: {layer_cls}; id= {count}')
        print(e)
        record_bug(count, 'wrong results', type(layer).__name__, 'wrong result')
        return
    print("[Success] This test case passed!")


def compile_keras(cnt, model, input_shape, input_data, temp_model_dir):
    # [reference](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/101-tensorflow-classification-to-openvino/101-tensorflow-classification-to-openvino.ipynb#scrollTo=uVXHcr4K7gS_)
    ov_model = ov.convert_model(model,  input=input_shape)
    ir_path = f"{temp_model_dir}/_temp_OVIR_{cnt}.xml"  # file must ends with 'xml'
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    core = ov.Core()
    model = core.read_model(ir_path)

    compiled_model = core.compile_model(model=model, device_name='CPU')  # CPU,GPU,AUTO
    # compiled_model = core.compile_model(model=model, device_name=device.value)  # CPU,GPU,AUTO

    # show the model structure
    # input_key = compiled_model.input(0)
    output_key = compiled_model.outputs
    # print("output_key:", output_key)
    result = []
    for output in output_key:
        result.append(compiled_model(input_data)[output])
    return result


if __name__ == '__main__':
    # pass
    # layer_test(keras.layers.ZeroPadding2D,args=(),kwargs={'padding':[4, 2],'data_format':"channels_last",},input_shape=[None, 112, 112, 96],input_dtype='float32',)
    # layer_test(keras.layers.Activation, kwargs={'activation': "relu", }, input_shape=[1, 2, 3, 4], )
    # layer_test(keras.layers.GlobalAveragePooling1D,args=(),kwargs={},input_shape=[12, 2, 11],input_dtype='uint8',)
    # layer_test(keras.layers.ReLU, args=(), kwargs={}, input_shape=[None, 4, 8, 3, 2], input_dtype='float32', )
    # layer_test(keras.layers.Attention,args=(),kwargs={'dropout':0.8,'use_scale':False,'score_mode':"dot",},input_shape=[2, 1, 4],input_dtype='float32',)
    # layer_test(keras.layers.RepeatVector, args=(2,), kwargs={}, input_shape=[3, 1], input_dtype='float32')
    # layer_test(keras.layers.UpSampling1D, args=(), kwargs={}, input_shape=[2, 1, 18], input_dtype='float64',
    #            count=30453)
    # layer_test(keras.layers.ZeroPadding2D, args=(), kwargs={'padding': [3, 1], 'data_format': "channels_last", },
    #            input_shape=[None, 150, 150, 144], input_dtype='float32', )
    # layer_test(keras.layers.Cropping3D,args=(),kwargs={'cropping':[1, 1, 1],'data_format':"channels_last",},input_shape=[2, 8, 8, 8, 2],input_dtype='float32',)
    # layer_test(keras.layers.ZeroPadding2D, kwargs={'padding': [4, 4], 'data_format': "channels_last", },
    #            input_shape=[None, 42, 42, 336], )
    # for i in range(22):
    #     layer_test(keras.layers.LeakyReLU,args=(),kwargs={'alpha':-5.609126659975927e+37,},input_shape=[15, 20, 9, 1],input_dtype='int16',count=26380,)
    # layer_test(keras.layers.BatchNormalization, args=(),
    #            kwargs={'epsilon': -1e+40, }, input_shape=[5, 1, 11, 1], input_dtype='float32', )
    layer_test(keras.layers.BatchNormalization, args=(), kwargs={},
               input_shape=[1, 2, 3, 4, 5], input_dtype='float32', )
