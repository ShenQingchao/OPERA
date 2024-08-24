import torch
import random
random.seed(20231122)


def input_dict2data(input_dict):
    input_shape = input_dict['shape']
    input_dtype = input_dict['dtype']

    # handle for example: torch.randn(None, dtype=[int,int,int]):
    if 'int,int,' in input_dtype:
        shape_dim = ''.count(input_dtype, "int")
        input_shape = []
        for i in range(shape_dim):
            input_shape.append(random.randint(1, 100))
        input_dtype = "NoneType"

    if input_dtype.startswith('torch.int') or input_dtype.startswith('torch.uint'):
        return f"torch.randint(1, 100, {input_shape}, dtype={input_dtype})"
    elif input_dtype == 'Tensor' or input_dtype == 'NoneType' or not input_dtype:
        return f"torch.randn({input_shape}, dtype=torch.float32)"
    elif input_dtype == 'torch.bfloat16':
        return f"torch.randn({input_shape}, dtype=torch.float16)"
    elif input_dtype == 'torch.bool':
        return f"torch.randint(0, 2, {input_shape}).bool()"
    else:
        # [tensor(1), tensor(2), tensor(3), tensor(4)]  -> [1, 2, 3, 4]
        if isinstance(input_shape, list) and len(input_shape) >= 1 and isinstance(input_shape[0], torch.Tensor):
            input_shape = [i.item() for i in input_shape]
        return f"torch.randn({input_shape}, dtype={input_dtype})"


def gen_fun_call_cls(api, para):
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


def gen_fun_call_func(api, para):
    api_name = api.split('.')[-1]

    params_list_declare_str = ""
    params_list_fill_str = ""
    params_list_fill_str_kv = ""
    params_list_no_key = []

    for k, v in para.items():  # parse value first, the parse the dict k=v. e.g., (2, 3,4 , p=1.4, threshold=0.1)
        if k.startswith("parameter:"):
            param_id = k.split("parameter:")[-1]
            params_list_no_key.append(int(param_id))

            if v == 'torchTensor':  # skip the test case, because, we cannot get the shape for the tensor.
                return None
            if isinstance(v, dict) and len(v) == 2 and 'shape' in v.keys() and 'dtype' in v.keys():  # tensor
                tensor_v = input_dict2data(v)
                params_list_declare_str += f"para_{param_id} = {tensor_v}\n"
            else:
                params_list_declare_str += f"para_{param_id} = "
                if isinstance(v, str):
                    if v.startswith('torch.nn.'):
                        params_list_declare_str += f"{v}\n"
                    else:
                        if len(v) >= 30:  # wrong/meaningless string.
                            v = ''
                        params_list_declare_str += f"'{v}'\n"
                else:
                    params_list_declare_str += f"{v}\n"
        elif k == 'input_signature':
            input_signature = v
        elif k == 'output_signature':
            output_signature = v
            pass
        # TODO: check the following branch
        elif k == 'input' and isinstance(v, dict) and len(v) == 2 and 'shape' in v.keys() and 'dtype' in v.keys():
            tensor_v = input_dict2data(v)
            params_list_declare_str += f"para_0 = {tensor_v}\n"
            params_list_no_key.append(0)
        else:  # k = v
            if isinstance(v, str):
                if v == 'torchdtype':
                    params_list_fill_str_kv += f"{k}=None,"
                else:
                    params_list_fill_str_kv += f"{k}='{v}',"
            else:
                params_list_fill_str_kv += f"{k}={v},"

    sorted_params_list_no_key = sorted(params_list_no_key)
    # assert len(sorted_params_list_no_key) == int(sorted_params_list_no_key[-1])
    for pa in sorted_params_list_no_key:
        if pa == 0:  # skip the input data.
            continue
        params_list_fill_str += f"para_{pa},"

    params_list_fill_str += params_list_fill_str_kv

    clazz_template = f"{params_list_declare_str}"
    clazz_template += f"class {api_name}(Module):\n"
    clazz_template += f"    def forward(self, *args):\n"
    clazz_template += f"        return {api}(args[0], {params_list_fill_str})\n"
    clazz_template += f"verify_model({api_name}().float().eval(), input_data=para_0)\n\n"
    return clazz_template


def record_op(func_name, params, input_signature, output_signature, output_file):
    params = dict(params)
    out_fname = "torch." + func_name
    params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    # print('debug:', out_fname, params)
    if func_name.startswith("nn.functional"):
        api_call = gen_fun_call_func(out_fname, params)
    else:
        api_call = gen_fun_call_cls(out_fname, params)

    from .utils import GlobalVar
    all_api_call = GlobalVar.all_api_call

    if api_call not in all_api_call:
        count = GlobalVar.count
        GlobalVar.add_count()

        with open(output_file, 'a') as f:
            f.write(f"# test_id: {count} \n{api_call}\n")
            # all_api_call.append(api_call)
            GlobalVar.add_call(api_call)

