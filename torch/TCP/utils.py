import inspect
import copy
import torch


def is_correct_api(api_str: str):
    module_list = api_str.split('.')

    module_name = module_list[0]
    for sub_module in module_list[1:]:
        if not hasattr(eval(module_name), sub_module):
            return False
        module_name = module_name + '.' + sub_module
    return True


def get_default_args_dict(func):
    argspec = inspect.getfullargspec(func.__init__)
    defaults = argspec.defaults
    args = argspec.args[1:]
    default_args = {}
    if len(args) != 0:
        defaults_len = len(defaults) if defaults else 0
        no_default_vars = args[:len(args) - defaults_len]

        if defaults:
            for arg, default in zip(reversed(args), reversed(defaults)):
                if arg == 'input':  # just for pytorch?
                    continue
                default_args[arg] = default
        if len(no_default_vars) != 0:
            for var in reversed(no_default_vars):
                if var == 'input':  # just for pytorch?
                    continue
                default_args[var] = 'no_default'
    # add input_shape as a param
    default_args['input_dtype'] = 'no_default'
    default_args['input_shape'] = 'no_default'

    default_args = dict(reversed(default_args.items()))
    return default_args


def preprocess_params(default_dict: dict, collected_dict: dict):
    result_dict = copy.deepcopy(default_dict)
    # import pdb;pdb.set_trace()
    # change the default_dict according to the collected real_para_value.
    ordered_key_list = list(default_dict.keys())
    for k, v in collected_dict.items():
        if k.startswith("para_"):
            para_id = int(k.split("para_")[-1]) + 2  # [warning] change it when add new attribute.
            result_dict[ordered_key_list[para_id]] = v
        elif k in result_dict.keys():
            result_dict[k] = v
        else:
            print(">>> [Warning] the key {k} is invisible in default_dict: {default_dict}")
            continue
            # print(f"")  # redundant para for **kwargs
    if 'no_default' in result_dict.values():  # the collected test case lack para value for the undefault para
        print('[debug] un-default para, ', result_dict)
        return False
    return result_dict


if __name__ == '__main__':
    import numpy

    res = is_correct_api('numpy.random.rand')
    print(res)
    res = get_default_args_dict(torch.nn.functional.relu6)
    print(res)
