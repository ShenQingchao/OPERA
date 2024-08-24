import inspect
import copy

import onnx
import importlib
#from onnx.reference.ops import op_resize
import re
pattern = r'([A-Z])'
def replace(match):
    return '_' + match.group(1).lower()

def is_correct_operator(operator_name: str):
    all_operators = onnx.defs.get_all_schemas()
    return any(op.name == operator_name for op in all_operators)

def is_correct_api(api_str: str):
    module_list = api_str.split('.')

    module_name = module_list[0]
    for sub_module in module_list[1:]:
        if not hasattr(eval(module_name), sub_module):
            return False
        module_name = module_name + '.' + sub_module
    return True


def get_onnx_op_default_args_dict(op_name):
    # Get the ONNX operator schema
    schema = onnx.defs.get_schema(op_name)
    default_args = {}
    # Get the attributes of the operator
    attributes = schema.attributes
    # Iterate over the attributes and add them to the default arguments dictionary
    for attr_name in attributes:
        attr = schema.attributes[attr_name]
        # Check if the attribute has a default value
        if not attr.default_value:
            default_args[attr_name] = 'no_default'
        else:
            try:
                default_value = onnx.helper.get_attribute_value(attr.default_value)
            except ValueError:
                if attr.required:
                    default_value = 'no_default'
                else:
                    default_value = None 
            default_args[attr_name] = default_value
    # Add input and output as params
    default_args['input_shape'] = 'no_default'
    default_args['input_dtype'] = 'no_default'
    default_args['output_dtype'] = 'no_default'
    default_args['output_shape'] = 'no_default'

    return default_args


#def get_onnx_op_default_args_dict(op_name):
#    special_list = ['TopK', 'ScatterND', 'MatMul', 'IsNaN', 'IsInf', 'DFT', 'EyeLike', 'Col2Im', 'BitShift', 'ArgMax', 'ArgMin', 'GatherND', 'GRU', 'LRN', 'LSTM', 'PRelu', 'RNN', 'STFT']
#    if op_name in special_list:
#        new_op_name = "_" + op_name.lower()
#    elif op_name == "MatMulInteger":
#        new_op_name = "_matmul_integer"
#    elif op_name == "QLinearConv":
#        new_op_name = "_qlinear_conv"
#    elif op_name == "QLinearMatMul":
#        new_op_name = "_qlinear_matmul"
#    elif op_name == "TfIdfVectorizer":
#        new_op_name = "_tfidf_vectorizer"
#    else:
#        new_op_name = re.sub(pattern, replace, op_name)
#    op_class_name = "op" + new_op_name
#    module_name = "onnx.reference.ops." + op_class_name
#    module = importlib.import_module(module_name)
#    # Check if the op_class exists in the module
#    if hasattr(module, op_name):
#        op_class = getattr(module, op_name)
#    else:
#        # If not, check for version specific classes
#        onnx_version = onnx.__version__
#        onnx_version = onnx_version[2:].split(".")[0]
#        print("onnx_version:", onnx_version)
#        version_specific_class_names = [op_name + "_" + str(i) for i in reversed(range(1, int(onnx_version) + 1))]
#        for class_name in version_specific_class_names:
#            if hasattr(module, class_name):
#                print("class_name", class_name)
#                op_class = getattr(module, class_name)
#                break
#        else:
#            raise ValueError(f"No suitable class found for operator {op_name} in ONNX version {onnx_version}")
#    #op_class = getattr(module, op_name)
#    #op_class = eval(op_name)
#    argspec = inspect.getfullargspec(op_class._run)
#    defaults = argspec.defaults
#    args = argspec.args[1:]
#    default_args = {}
#    if len(args) != 0:
#        defaults_len = len(defaults) if defaults else 0
#        no_default_vars = args[:len(args) - defaults_len]
#
#        if defaults:
#            for arg, default in zip(reversed(args), reversed(defaults)):
#                default_args[arg.lower()] = default
#        if len(no_default_vars) != 0:
#            for var in reversed(no_default_vars):
#                default_args[var.lower()] = 'no_default'
#    # add output_shape as a param
#    default_args['output_name'] = 'no_default'
#    default_args['output_dtype'] = 'no_default'
#    default_args['output_shape'] = 'no_default'

#    default_args = dict(reversed(default_args.items()))
    #default_args.pop('x')
#    return default_args


def get_default_args_dict(func):
    argspec = inspect.getfullargspec(func.__init__)
    print(func, argspec)
    defaults = argspec.defaults
    args = argspec.args[1:]
    default_args = {}
    if len(args) != 0:
        defaults_len = len(defaults) if defaults else 0
        no_default_vars = args[:len(args) - defaults_len]

        if defaults:
            for arg, default in zip(reversed(args), reversed(defaults)):
                default_args[arg] = default
        if len(no_default_vars) != 0:
            for var in reversed(no_default_vars):
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
            print("=========", k, v)
            print(f">>> [Warning] the key {k} is invisible in default_dict: {default_dict}")
            continue
            # print(f"")  # redundant para for **kwargs
    if 'no_default' in result_dict.values():  # the collected test case lack para value for the undefault para
        print('[debug] un-default para, ', result_dict)
        return False
    return result_dict


if __name__ == '__main__':
    res = is_correct_operator('LinearRegressor')
    print(res)
    op_class_name = 'LinearRegressor'  # or 'Abs', 'Acos', etc.
    default_args = get_onnx_op_default_args_dict(op_class_name)
    print(default_args)

