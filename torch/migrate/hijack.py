import torch.nn.utils.prune
import inspect
from .decorate_func import decorate_function
from .decorate_class import decorate_class


def hijack_api(obj, func_name_str, mode, output_file):
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]

    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    def is_class(x):
        return inspect.isclass(x)

    def is_callable(x):
        return callable(x)

    if mode == "function":
        wrapped_func = decorate_function(orig_func, func_name_str, output_file)
    elif mode == "class":  # deed code
        wrapped_func = decorate_class(orig_func, func_name_str, output_file)
    else:
        if is_class(orig_func):
            wrapped_func = decorate_class(orig_func, func_name_str, output_file)
        elif is_callable(orig_func):
            wrapped_func = decorate_function(orig_func, func_name_str, output_file)
        else:
            wrapped_func = orig_func
    setattr(module_obj, func_name, wrapped_func)


def hijack(output_file):
    with open(__file__.replace("hijack.py", "operator_list.txt"), "r") as f2:
        lines = f2.readlines()
        for api in lines:
            api = api.strip()
            if api.startswith("nn.functional."):
                hijack_api(torch, api, "function", output_file=output_file)
            else:
                hijack_api(torch, api, "", output_file=output_file)
    print(f"finish instrumentation for total {len(lines)} operators")
