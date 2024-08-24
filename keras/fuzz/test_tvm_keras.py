import numpy as np
from tensorflow.keras import layers, models
import tvm
import tvm.relay as relay
import traceback
import re
import logging

logging.basicConfig(level=logging.ERROR)
np.random.seed(2023)


def extract_crash_message(e):
    tb = traceback.extract_tb(e.__traceback__)
    file_name, line_number, _, _ = tb[-1]
    file_name = file_name[len("/workplace/software/tvm/tvm_/"):]
    exc_type = type(e).__name__
    stack_trace = str(e).strip().split("\n")[-1]
    if stack_trace.endswith(':'):
        stack_trace = stack_trace[:-1]
    stack_trace = stack_trace.split(':')[-1].strip()
    pattern = r"[\[\(].*?[\]\)]"
    stack_trace = re.sub(pattern, "", stack_trace)
    print(f">>>>>>>>>>>>>>>>>>>Bug Info: {stack_trace}")
    # exc_value = str(e)
    # exception_info = traceback.format_exception_only(exc_type, exc_value)
    # crash_message = f"{exception_info}_{file_name}_{line_number}"

    crash_message = f"{exc_type}_{file_name}_{line_number}_{stack_trace}"
    return crash_message


def record_bug(bug_id, bug_type, op, crash_message=''):
    bug_info_str = f"{bug_id}\t{bug_type}\t{op}\t{crash_message}\n"

    with open("../data/detected_bugs_tvm.txt", 'a', encoding='utf-8') as f:
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


def correct_shape(input_shape, layer_name):
    if input_shape is None:
        if '3D' in layer_name:
            return [np.random.randint(1, 32) for i in range(5)]
        elif '1D' in layer_name:
            return [np.random.randint(1, 32) for i in range(3)]
        return [np.random.randint(1, 32) for i in range(4)]  # default input_shape in 4 dim

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

        # parse layout
        data_format = kwargs["data_format"] if "data_format" in kwargs.keys() else None

        input_shape = correct_shape(input_shape, layer_cls.__name__)

        if data_format:
            input_layout = "NCHW" if data_format == "channels_first" else "NHWC"
            if len(input_shape) == 5:  # solve the conflict constraints
                input_shape = input_shape[1:]
        elif len(input_shape) == 5:
            input_layout = "NDHWC"
        elif len(input_shape) == 3:
            input_layout = "NWC"
        else:
            input_layout = "NHWC"  # default for keras model

        input_shape = parse_input_shape(input_shape)

        input_data = assign_input_data(input_shape, input_dtype)  # Notice, input_shape will be changed.
        # input_shape = input_data.shape  # update input_shape

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
    except Exception as e:
        print("[keras error]", e)
        return
    try:
        res_tvm = compile_keras(model, input_shape, input_data, dtype=input_dtype, exec_mod='vm',
                                input_layout=input_layout)
    except Exception as e:
        if 'support' not in str(e):
            print(f'[bug in tvm] using test: {layer_cls}; id= {count}')
            print(e)
            crash_message = extract_crash_message(e)
            record_bug(count, 'crash', type(layer).__name__, crash_message=crash_message)
        return
    try:
        if isinstance(res_keras, list):  # multiple output:
            for i in range(len(res_keras)):
                np.testing.assert_allclose(res_keras[i], res_tvm[i], atol=1e-3, rtol=1e-3)
        else:
            np.testing.assert_allclose(res_keras, res_tvm, atol=1e-3, rtol=1e-3)
    except AssertionError as e:
        print(f'[bug in tvm] using test: {layer_cls}; id= {count}')
        print(e)
        record_bug(count, 'wrong results', type(layer).__name__, 'wrong result')
        return
    print("[success] This test case passed!")

    # ----------------------test as first layer in Sequential API---------------------------------------
    # ---------------------- generate sequential model which diff from function model ------------------
    '''
    try:
        layer_config = layer.get_config()
        layer_config["batch_input_shape"] = input_shape
        layer = layer.__class__.from_config(layer_config)

        model = models.Sequential()
        model.add(layers.Input(shape=input_shape[1:], dtype=input_dtype))
        model.add(layer)

        layer_weights = (
            layer.get_weights()
        )
        layer.set_weights(layer_weights)
        # res_keras = model.predict(input_data)
        res_keras = model(input_data)
        # model.summary()
    except Exception as e:
        print(e)
        return
    try:
        res_tvm = compile_keras(model, input_data.shape, input_data, dtype=input_dtype, exec_mod='aot',
                                input_layout=input_layout)
    except Exception as e:
        if 'support' not in str(e):  # skip the unsupported behavior
            print(f'[bug in tvm2][seq_graph][aot_mod] using test: {layer_cls}; id = {count}')
            print(e)
            crash_message = extract_crash_message(e)
            record_bug(count, 'crash', type(layer).__name__, crash_message=crash_message)
        return
    try:
        # print(np.shape(res_tvm), np.shape(res_keras))
        np.testing.assert_allclose(res_keras, res_tvm, atol=1e-3, rtol=1e-3)
    except AssertionError as e:
        print(f'[bug in tvm2][seq_graph][aot_mod] using test: {layer_cls}; id = {count}')
        print(e)
        record_bug(count, 'wrong results', type(layer).__name__, 'wrong result')
        return
    return'''


def compile_keras(model, input_shape, input_data, dtype='float32', exec_mod='graph', input_layout="NCHW"):
    target = 'llvm'
    ctx = tvm.cpu(0)

    # input_name = model.input.name.split(':')[0]
    # shape_dict = {input_name: input_shape}
    if isinstance(input_shape[0], list):  # multiple inputs
        shape_dict = {name: x.shape for (name, x) in zip(model.input_names, input_data)}
    else:
        shape_dict = {model.input_names[0]: input_shape}
    # print(">>>>> shape_dict:", shape_dict)
    irmod, params = relay.frontend.from_keras(model, shape_dict, layout=input_layout)
    # print(irmod)
    with tvm.transform.PassContext(opt_level=3):
        model = relay.build_module.create_executor(exec_mod, irmod, ctx, target, params).evaluate()  # graph, vm, aot

    if isinstance(input_shape[0], list):
        test_x_tvm = []
        for input_i in input_data:
            test_x_tvm.append(tvm.nd.array(input_i.astype(dtype)))
        tvm_out = model(*test_x_tvm)
    else:
        test_x_tvm = tvm.nd.array(input_data.astype(dtype))
        tvm_out = model(test_x_tvm)
    if isinstance(tvm_out, list):  # multiple output
        tvm_out_np = []
        for out_i in tvm_out:
            tvm_out_np.append(out_i.numpy())
    else:
        tvm_out_np = tvm_out.numpy()  # single output

    return tvm_out_np


if __name__ == '__main__':
    pass
