import copy
import re
import inspect
from collections import OrderedDict

import torch

import case
import utils
from case import TC, TCDict
from utils import is_correct_api


def get_default_args_dict_torch_func(func_name):
    # 对于torch的两种类型的类型cls和func， 获取默认参数的方式不同，对于cls的获取方法和Keras的方式相同utils.get_default_args_dict()相同，
    # 对于func的API，获取方式如下，该方法存在很多参数获取不到，所以需要对获取不到的参数进行特殊处理，
    # # TODO：@ruifeng，检查该函数的正确性

    default_args = {}
    if func_name == 'torch.nn.functional.one_hot':
        default_args['tensor'] = 'no_default'
        default_args['num_classes'] = -1
        default_args['input_dtype'] = 'no_default'
        default_args['input_shape'] = 'no_default'
        default_args = dict(reversed(default_args.items()))
        return default_args
    if func_name == 'torch.nn.functional.pad':
        default_args['input'] = 'no_default'
        default_args['pad'] = 'no_default'
        default_args['mode'] = 'constant'
        default_args['value'] = None
        default_args['input_dtype'] = 'no_default'
        default_args['input_shape'] = 'no_default'
        default_args = dict(reversed(default_args.items()))
        return default_args
    elif func_name == 'torch.nn.functional.pdist':
        default_args['input'] = 'no_default'
        default_args['p'] = 2
        default_args['input_dtype'] = 'no_default'
        default_args['input_shape'] = 'no_default'
        default_args = dict(reversed(default_args.items()))
        return default_args

    try:
        signature = inspect.signature(eval(func_name))
    except Exception as e:
        # print(f"Failed when get the signature of {func_name}")
        # return utils.get_default_args_dict(eval(func_name))
        suffix = func_name[len('torch.nn.functional.'):]
        func_name = 'torch.nn.' + suffix.replace("_", " ").title().replace(" ", "")
        func_name = func_name.replace('Prelu', 'PReLU')
        func_name = func_name.replace('Rrelu', 'RReLU')
        func_name = func_name.replace('Relu', 'ReLU')
        func_name = func_name.replace('Elu', 'ELU')
        func_name = func_name.replace('Glu', 'GLU')
        func_name = func_name.replace('Celu', 'CELU')
        func_name = func_name.replace('Gelu', 'GELU')
        func_name = func_name.replace('Selu', 'SELU')
        func_name = func_name.replace('Silu', 'SiLU')
        func_name = func_name.replace('Logsigmoid', 'LogSigmoid')
        func_name = func_name.replace('3D', '3d')
        func_name = func_name.replace('2D', '2d')
        func_name = func_name.replace('1D', '1d')
        func_name = func_name.replace('BatchNorm', 'BatchNorm1d')
        func_name = func_name.replace('InstanceNorm', 'InstanceNorm1d')
        func_name = func_name.replace('BatchNorm1d1d1d', 'BatchNorm1d')
        func_name = func_name.replace('InstanceNorm1d1d1d', 'InstanceNorm1d')

        # print(f'func_name : {func_name}')
        if is_correct_api(func_name):
            return utils.get_default_args_dict(eval(func_name))

        print(f"Failed when get the signature of {func_name}")
        return None

    # 获取参数字典
    parameters = signature.parameters
    # 遍历参数字典，获取参数名称和默认取值
    default_args['input_shape'] = 'no_default'
    default_args['input_dtype'] = 'no_default'
    for name, parameter in parameters.items():
        default = parameter.default
        if default == inspect._empty:
            default = "no_default"
        if name == 'input':
            continue

        default_args[name] = default
        # print(f"Parameter name: {name}")
        # print(f"Default value: {default}")
    # default_args = dict(reversed(default_args.items()))
    if 'kwargs' in default_args:
        suffix = func_name[len('torch.nn.functional.'):]
        func_name = 'torch.nn.' + suffix.replace("_", " ").title().replace(" ", "")
        func_name = func_name.replace('Prelu', 'PReLU')
        func_name = func_name.replace('Relu', 'ReLU')
        func_name = func_name.replace('Elu', 'ELU')
        func_name = func_name.replace('Glu', 'GLU')
        func_name = func_name.replace('Celu', 'CELU')
        func_name = func_name.replace('Gelu', 'GELU')
        func_name = func_name.replace('Selu', 'SELU')
        func_name = func_name.replace('Silu', 'SiLU')
        func_name = func_name.replace('Logsigmoid', 'LogSigmoid')
        func_name = func_name.replace('3D', '3d')
        func_name = func_name.replace('2D', '2d')
        func_name = func_name.replace('1D', '1d')
        func_name = func_name.replace('BatchNorm', 'BatchNorm1d')
        func_name = func_name.replace('InstanceNorm', 'InstanceNorm1d')
        func_name = func_name.replace('BatchNorm1d1d1d', 'BatchNorm1d')
        func_name = func_name.replace('InstanceNorm1d1d1d', 'InstanceNorm1d')

        # print(f'func_name : {func_name}')
        return utils.get_default_args_dict(eval(func_name))
    return default_args


def args2list(args):
    args = args.split(',')[:-1]
    status = [0, 0, 0]  # (, [, {
    argv = []
    tmp = ''
    cnt = 0
    # 解析参数，需要处理list，tuple。
    for arg in args:
        arg = arg.strip()
        if tmp == '':
            tmp = arg
        else:
            tmp += ', ' + arg
        status[0] += len(re.findall('\(', arg)) - len(re.findall('\)', arg))
        status[1] += len(re.findall('\[', arg)) - len(re.findall('\]', arg))
        status[2] += len(re.findall('\{', arg)) - len(re.findall('\}', arg))
        if status == [0, 0, 0]:
            argv.append(tmp)
            tmp = ''
    return argv


def parse_pytorch_func(test_cmd_str):
    # para_0 = torch.randn([20], dtype=torch.float64)
    # class softmax(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.softmax(args[0], dim=-1, )
    # verify_model(softmax().float().eval(), input_data=para_0)
    all_para_info = {}
    lines = test_cmd_str.split('\n')
    for line in lines:
        line = replace_tensors(line)
        if line.startswith('para_'):  # normal parameter
            arg, val = line.split(' = ')  # ['para1', '1']
            if val.find('torch.Size') != -1:
                val = val[len('torch.Size('): -1]
                all_para_info[arg] = eval(val)
            elif arg == 'para_0':
                if val.find('torch.randn') != -1:
                    _, input_str = val.split('torch.randn(')
                    input_shape, input_dtype = input_str.strip()[:-1].split(", dtype=")
                    input_shape = eval(input_shape)
                    all_para_info['input_shape'] = input_shape
                    all_para_info['input_dtype'] = input_dtype
                elif val.find('torch.randint') != -1:
                    _, input_str = val.split('torch.randint(')
                    input_str = input_str[input_str.find(',', 2) + 1:]
                    input_shape, input_dtype = input_str.strip()[:-1].split(", dtype=")
                    input_shape = eval(input_shape)
                    all_para_info['input_shape'] = input_shape
                    all_para_info['input_dtype'] = input_dtype
                else:
                    raise NotImplemented
            elif val.find('torch.rand') == -1:
                all_para_info[arg] = eval(val)
            else:  # torch.randn()/torch.randint()
                if val.find('torch.randn') != -1:
                    _, input_str = val.split('torch.randn(')
                    input_shape, input_dtype = input_str.strip()[:-1].split(", dtype=")
                    input_shape = eval(input_shape)
                    all_para_info[arg] = input_shape
                    # all_para_info[arg + '_dtype'] = input_dtype
                elif val.find('torch.randint') != -1:
                    _, input_str = val.split('torch.randint(')
                    input_str = input_str[input_str.find(',', 2) + 1:]
                    input_shape, input_dtype = input_str.strip()[:-1].split(", dtype=")
                    input_shape = eval(input_shape)
                    # all_para_info[arg + '_dtype'] = input_dtype
                else:
                    raise NotImplemented
        elif line.find('return torch.nn.functional') != -1:
            layer = line[line.find('torch.nn.functional'):line.find('(')]
            args = line[line.find('(') + 1:-1]
            argv = args2list(args)
            for arg in argv:
                arg = arg.strip()
                if arg == 'arg[0]' or arg == '':
                    continue
                if arg.startswith('para_'):
                    continue
                if arg.find('=') != -1:
                    # print(f'arg {arg}')
                    key = arg.split('=')[0]
                    val = arg[arg.find('=')+1:]
                    # print(f'val {val}')
                    if key == 'input':
                        shape, dtype = val.strip()[1:-1].split(", 'dtype': ")
                        _, shape = shape.split("'shape': ")
                        all_para_info['input_shape'] = eval(shape)
                        all_para_info['input_dtype'] = dtype
                    else:
                        all_para_info[key] = eval(val)

    if 'input_dtype' in all_para_info and all_para_info['input_dtype'].find('Tensor') != -1:
        # print(f'Invalid func: {test_cmd_str}')
        return "Invalid", "no_default"

    return layer, all_para_info


def parse_torch_cls(test_cmd_str):
    # verify_model(torch.nn.Linear(3,3,bias=False,).eval(), input_data=[torch.randn([3], dtype=torch.float32)])
    test_cmd_str = replace_tensors(test_cmd_str)

    all_para_info = {}
    layer_args, input_str = test_cmd_str.split(").eval(), input_data=[")
    input_str = input_str.strip()[:-1]
    input_str = input_str.split('torch.rand')[1:]
    if len(input_str) == 1:
        input_shape, input_dtype = input_str[0].strip()[:-2].split(", dtype=")
        if input_shape.startswith('n'):
            input_shape = input_shape[2:]
        elif input_shape.startswith('int'):
            input_shape = input_shape[input_shape.find(',', 2) + 1:]
        input_shape = eval(input_shape)
    else:
        input_shape = []
        input_dtype = []
        for input_torch in input_str:
            if input_torch.startswith('n'):
                input_torch = input_torch[2:-1]
                # print(f'input_torch {input_torch}')
                shape, dtype = input_torch[:-1].split(", dtype=")
                shape = eval(shape)
                input_shape.append(shape)
                input_dtype.append(dtype)
            elif input_torch.startswith('int'):
                input_torch = input_torch[4:-1]
                args, dtype = input_torch.split(", dtype")
                shape = args[args.find(',', 2) + 1:]
                shape = eval(shape)
                input_shape.append(shape)
                input_dtype.append(dtype)
            else:
                raise NotImplemented

    if str(input_dtype).find('Tensor') != -1:
        # print(f'Invalid: {test_cmd_str}')
        return "Invalid", "no_default"

    all_para_info['input_shape'] = input_shape
    all_para_info['input_dtype'] = input_dtype

    layer_args = layer_args[len("verify_model("):].strip()
    layer = layer_args.split("(")[0]
    args = layer_args[len(layer) + 1:]
    args = args.split(',')[:-1]
    tmp = ''
    cnt = 0
    # 解析参数，需要处理list，tuple。
    for arg in args:
        arg = arg.strip()
        if tmp == '':
            tmp = arg
        else:
            tmp += ', ' + arg
        if (tmp.find('[') != -1 and tmp.find(']') != -1) \
                or (tmp.find('(') != -1 and tmp.find(')') != -1) \
                or (tmp.find('[') == -1 and tmp.find('(') == -1):
            cnt += 1
            key = f'para_{cnt}'
            val = tmp
            if tmp.find('=') != -1:
                key, val = tmp.split('=')
            try:
                val = eval(val)
            except Exception as e:
                val = val

            all_para_info[key] = val
            tmp = ''
    # print(f'{test_cmd_str}, {layer}, {all_para_info}')

    return layer, all_para_info


def preprocess_torch_test(test_file):
    all_test_case_set = set()
    all_layer_default_args_dict = {}
    cnt = 0
    with open(test_file, 'r', encoding='utf-8') as test_f:
        all_lines = test_f.readlines()
    all_lines = ''.join(all_lines)
    pattern = r'# test_id: \d+ \n'
    all_test_cases = re.split(pattern, all_lines)
    # print(all_test_cases)
    all_test_cases = all_test_cases[1:]
    tc_dict = TCDict()
    new_tests = []
    layers = set()
    for test_case in all_test_cases:
        is_valid = False
        test_case = test_case.strip()
        if test_case in all_test_case_set:  # deduplicate
            continue
        all_test_case_set.add(test_case)
        cnt += 1
        if test_case.startswith("verify_model"):
            # print(f'testcase: {test_case}')
            layer, all_para_info = parse_torch_cls(test_case)
        elif test_case.startswith("para") or test_case.startswith("class"):
            layer, all_para_info = parse_pytorch_func(test_case)
        else:
            print(f'not implemented {test_case}')
            raise NotImplemented

        if layer == 'Invalid':
            continue

        layers.add(layer)
        if is_correct_api(layer):
            if layer not in case.all_layer_default_args_dict.keys():
                if layer.find('functional') != -1:
                    default_args_dict = get_default_args_dict_torch_func(
                        layer)
                else:
                    if layer == 'torch.nn.RNN' or layer == 'torch.nn.LSTM' or layer == 'torch.nn.GRU':
                        default_args_dict = utils.get_default_args_dict(eval('torch.nn.RNNBase'))
                    else:
                        default_args_dict = utils.get_default_args_dict(eval(layer))

                # get the complete para info using dict
                # the default args use abstract format
                if default_args_dict is None or 'kwargs' in default_args_dict:
                    continue
                default_args_dict = TC.get_abstract_test_case(default_args_dict)
                case.all_layer_default_args_dict[layer] = default_args_dict
                print(f'layer default: {layer} -- {default_args_dict}')
            else:
                default_args_dict = case.all_layer_default_args_dict[layer]

            # let each test case have the same para number
            # print(f'testcase {test_case}')
            processed_args = preprocess_params(default_args_dict, all_para_info)
            # print(f'DEBUG processed_args {processed_args}')
            # print('>>> default_args_dict:', default_args_dict)
            if processed_args:  # return False if lack value for an un-default para.
                all_para_info = processed_args
                abstract_tc = TC.get_abstract_test_case(all_para_info)
                # print('>>> processed_args:', abstract_tc)
                encode = abstract_tc.values()
                is_valid = True
                new_tc = TC(test_cmd_str=test_case + '\n', layer=layer, all_para_info=all_para_info,
                            abstract_tc=abstract_tc, encode=encode)
                tc_dict.add(new_tc)
            else:
                is_valid = False

        # print(f'TestCase: \n{test_case}\nlayer: {layer}\nabstract: {abstract_tc}')
    # case.all_layer_default_args_dict(all_layer_default_args_dict)
    index = 0
    for layer in layers:
        index += 1
        print('Layer ', index,  layer)
    return tc_dict


def replace_tensors(string):
    pattern = r'tensor\((\d+)\)'
    result = re.sub(pattern, r'\1', string)
    return result


def preprocess_params(default_dict: dict, collected_dict: dict):
    result_dict = copy.deepcopy(default_dict)
    # import pdb;pdb.set_trace()
    # change the default_dict according to the collected real_para_value.
    ordered_key_list = list(default_dict.keys())

    for k, v in collected_dict.items():
        if k.startswith("para_"):
            para_id = int(k.split("para_")[-1]) + 1  # [warning] change it when add new attribute.
            result_dict[ordered_key_list[para_id]] = v
        elif k in result_dict.keys():
            result_dict[k] = v
        else:
            # todo: these redundant params will lead to test case fail
            print(f'ordered_key_list = {ordered_key_list}')
            print(f'collected_list = {collected_dict}')
            print(f">>> [Warning] the key {k} is invisible in default_dict: {default_dict}")
            continue
            # print(f"")  # redundant para for **kwargs
    if 'no_default' in result_dict.values():  # the collected test case lack para value for the undefault para
        # print('[debug] un-default para, ', result_dict)
        return False
    return result_dict


if __name__ == '__main__':
    preprocess_torch_test(test_file="data/torch_borrow_all_test.py")
