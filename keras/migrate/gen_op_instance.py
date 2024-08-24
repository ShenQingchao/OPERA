import random


def get_tensor_shape(input_dict, input_dim=4):
    input_dtype = 'float32'

    default_shape = [3, 4, 5, 6]
    if input_dim == 3:
        default_shape = [2, 3, 4]
    elif input_dim == 5:
        default_shape = [2, 3, 4, 5, 6]

    label = input_dict['Label']
    if label == 'list':
        shape_value_str_list = input_dict['value']
        input_shape = []
        for elem in shape_value_str_list:
            all_list_types = ['KerasTensor', 'tensor', 'tf_object', 'variable', 'nparray', 'list']
            if elem['Label'] in all_list_types:
                if len(shape_value_str_list) == 1:
                    if 'shape' in elem.keys():
                        input_shape = elem['shape']
                    elif 'Label' in elem.keys() and elem['Label'] == 'list':
                        for sub_item in elem['value']:
                            input_shape.append(sub_item['shape'] if 'shape' in sub_item.keys() else random.randint(1, 4))
                    else:
                        print("[debug] input_dict:", input_dict)
                        input_shape = default_shape
                    # parse input_dtype
                    if 'dtype' in elem.keys():
                        input_dtype = elem['dtype']
                    else:
                        input_dtype = 'float32'
                else:
                    elem_input_shape_last_dim = elem['shape'][-1] if 'shape' in elem.keys() else random.randint(1, 4)
                    # a workaround: only support the single tensor as input.
                    input_shape.append(elem_input_shape_last_dim)
            elif elem['Label'] == 'raw' and elem['value'].split('-')[-1].isdigit():
                input_shape.append(int(elem['value']))
            elif elem['Label'] == 'other' and elem['type'] == "<class 'NoneType'>":
                input_shape.append(random.randint(1, 4))
            else:
                print("[debug2] input_dict:", input_dict)

        if len(input_shape) == 0:
            print("[debug3] input_dict:", input_dict)
            input_shape = default_shape

    elif label == 'tensor':
        if 'dtype' in input_dict.keys():
            input_dtype = input_dict['dtype']
        else:
            input_dtype = 'float32'

        if 'shape' in input_dict.keys():
            input_shape = input_dict['shape']
        else:
            print("[debug4] input_dict:", input_dict)
            input_shape = default_shape

    return list(input_shape), input_dtype


def gen_fun_call(api, para):
    temp_api = api.split('.')
    if len(temp_api) <= 4:
        return None
    if 'src.' in api:  # high tf version  e.g., keras.src.layers.activation.relu.ReLU
        api_name = f"keras.{temp_api[2]}.{temp_api[-1]}"
    elif api.startswith('keras.layers.'):
        api_name = f"keras.{temp_api[1]}.{temp_api[-1]}"
    else:  # low tf version e.g., tensorflow.python.keras.layers.advanced_activations.Softmax
        api_name = f"keras.{temp_api[3]}.{temp_api[-1]}"

    input_dim = 4
    if '1D' in api_name:
        input_dim = 3
    elif '3D' in api_name:
        input_dim = 5

    params_list_fill_str_kv = ""
    params_list_no_key_str = ""

    for k, v in para.items():  # parse the value without key.  e.g., (2, 3,4 , p=1.4, threshold=0.1)
        if k == 'trainable':
            continue
        if k.startswith("parameter:"):
            if v['Label'] == 'list' or v['Label'] == 'KerasTensor':
                p_shape, p_dtype = get_tensor_shape(v, input_dim)
                params_list_no_key_str += f"{p_shape},"
            else:
                if v['Label'] == 'other' and 'dtype' in v.keys() and 'None' in v['dtype']:
                    params_list_no_key_str += f"None,"
                else:
                    params_list_no_key_str += f"{v['value']}," if 'value' in v.keys() else ''
        elif k == 'dtype':
            continue
        elif k == 'input_signature':
            input_signature = v
            # print('>>>>>> input signature', v)
        elif k == 'output_signature':
            output_signature = v
            pass
        else:  # k = v
            if 'batch_input_shape' == k or 'input_shape' == k:  # parse it as input_shape rather than a parameter
                continue
            elif v['Label'] == 'list' or v['Label'] == 'KerasTensor':
                p_shape, _ = get_tensor_shape(v, input_dim)
                params_list_fill_str_kv += f"'{k}':{p_shape},"

            # special case handling: ELU {'alpha': {'Label': 'other', 'type': "<class 'NoneType'>"}}
            elif v['Label'] == 'other' and 'type' in v.keys() and 'NoneType' in v['type']:
                params_list_fill_str_kv += f"'{k}':None,"
            # ignore the value is an object.
            elif v['Label'] == 'other' or v['Label'] == 'tf_object':
                continue
            elif 'value' in v.keys() and '{"class_name":' in v['value']:
                continue
            elif type(v) == dict:
                continue
            else:
                params_list_fill_str_kv += f"'{k}':{v['value']}," if 'value' in v.keys() else ''

    if 'batch_input_shape' in para.keys():  # priority: L1
        input_shape, input_dtype = get_tensor_shape(para['batch_input_shape'], input_dim)
    elif 'input_shape' in para.keys():  # priority: L1
        input_shape, input_dtype = get_tensor_shape(para['input_shape'], input_dim)
        input_shape.insert(0, random.randint(1, 4))  # set batch_size = random
    elif input_signature:  # priority: L2
        input_shape, input_dtype = get_tensor_shape(input_signature, input_dim)
    else:  # priority: L3
        if '1D' in api_name:
            input_shape = [2, 3, 4]
        elif '3D' in api_name:
            input_shape = [2, 3, 4, 5, 6]
        else:
            input_shape = [2, 3, 4, 5]
        input_dtype = 'float32'

    clazz_template = f"layer_test({api_name},"
    clazz_template += f"args=({params_list_no_key_str}),"
    clazz_template += "kwargs={" + params_list_fill_str_kv + "},"
    clazz_template += f"input_shape={input_shape},input_dtype='{input_dtype}',)"
    # for some corner case:
    clazz_template = clazz_template.replace(":false,", ":False,").replace(":true,", ":True,")
    return clazz_template


all_api_call = []
count = 0


def record_op(func_name, params, input_signature, output_signature, output_file):
    # out_fname = f"tf.{func_name}"
    # print('debug', out_fname)
    if 'initializers' in func_name or 'engine' in func_name or 'OpLambda' in func_name or 'test' in func_name:
        return
    skip_op_list = ['BasicRNNCell', 'ResidualWrapper', 'DeviceWrapper', 'RandomFourierFeatures', 'DropoutWrapper',
                    'PeepholeLSTMCell', 'IndexLookup', 'Reduction', 'InstanceMethod', 'ClassMethod',
                    'premade', 'legacy_tf_layers', 'CustomLayerWithConfig', 'distribute']
    if func_name.split('.')[-1] in skip_op_list or 'optimizer_v2' in func_name:
        return

    params = dict(params)
    if input_signature:
        params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    # print('debug:', out_fname, params)

    api_call = gen_fun_call(func_name, params)
    if api_call:
        global count, all_api_call
        count += 1
        if api_call not in all_api_call:  # only save the deduplicate api_call
            all_api_call.append(api_call)
            with open(output_file, 'a', encoding='utf-8') as deduplicate_f:
                deduplicate_f.write(f"{api_call}\n")
