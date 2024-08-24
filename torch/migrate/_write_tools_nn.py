def input_dict2data(input_dict):
    res_data = 'torch.randn('
    input_shape = input_dict['shape']
    input_dtype = input_dict['dtype']
    if input_dtype.startswith('torch.int') or input_dtype.startswith('torch.uint'):
        return f"torch.randint(0, 10, {input_shape}, dtype={input_dtype})"
    elif input_dtype == 'Tensor' or input_dtype == 'NoneType' or not input_dtype:
        return f"torch.randn({input_shape}, dtype=torch.float32)"
    elif input_dtype == 'torch.bfloat16':
        return f"torch.randn({input_shape}, dtype=torch.float16)"
    elif 'int,int,' in input_dtype:  # handle for example: torch.randn(None, dtype=[int,int,int])
        return f"torch.randn({input_shape}, dtype=torch.int32)"
    else:
        return f"torch.randn({input_shape}, dtype={input_dtype})"


def gen_fun_call(api, para):
    api_call = "verify_model(" + api + '('
    tensor_var_num = 0
    for k, v in para.items():
        if isinstance(v, dict) and k != 'output_signature':
            if len(v) == 2 and 'shape' in v.keys() and 'dtype' in v.keys():  # input_data
                tensor_v = input_dict2data(v)
                api_call = f"input_data_{tensor_var_num}=" + tensor_v + "\n" + api_call
                api_call += f"input_data_{tensor_var_num},"
                tensor_var_num += 1

        elif k.startswith('parameter'):
            if v != 'torchTensor':
                if isinstance(v, str):
                    if v.startswith('torch.nn.'):
                        api_call += f"{v},"
                    else:
                        api_call += f"'{v}',"
                else:
                    api_call += f"{v},"
        elif k == 'input_signature':
            input_signature = v
        elif k == 'output_signature':
            output_signature = v
        else:
            if isinstance(v, str):
                api_call += f"{k}='{v}',"
            else:
                api_call += f"{k}={v},"
    api_call += ')'
    if 'functional' not in api:
        api_call += '.eval()'

    api_call += ', input_data=['
    if input_signature:
        for input_dict in input_signature:
            api_call += f"{input_dict2data(input_dict)},"
        api_call = api_call[:-1]
    elif "input_data_0" in api_call:
        api_call += "input_data_0"
    else:
        print("[warning] no input data is collected!")

    api_call += "])"
    return api_call


all_api_call = []


def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "torch." + func_name
    params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    print(out_fname, params)
    api_call = gen_fun_call(out_fname, params)
    with open('./borrow_func.py', 'a') as f:
        global all_api_call
        if api_call not in all_api_call:
            f.write(f"{api_call}\n")
            all_api_call.append(api_call)
    # print('debug:', out_fname, params)