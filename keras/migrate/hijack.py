import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import inspect
from tensorflow.migrate.decorators import dump_signature_of_class, dump_signature_of_function


def hijack_api(obj, func_name_str, output_file):
    """
    Args:
        obj: the base module. This function is currently specific to TensorFlow.
        func_name_str: A string. The full name of the api (except 'tf.'). For example, the name of
            `tf.keras.activations.elu` should be 'keras.activations.elu'.

    Returns:
        A boolean, indicating if hijacking is successful.

    The targeted API can be either a function or a class (the type will be detected by this function).
    This function would replace the original api with the new decorated api we created. This is achieved
    in a fairly simple and straight-forward way. For the example above, we just set the attribute by calling
    `setattr(tf.keras.activations, 'elu', wrapped_func)`.
    """
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]

    # Get the module object and the api object.
    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    # Utilities.
    def is_class(x):
        return inspect.isclass(x)

    def is_callable(x):
        return callable(x)

    def is_built_in_or_extension_type(x):
        if is_class(x) and hasattr(x, '__dict__') and not '__module__' in x.__dict__:
            return True
        else:
            return False

    # Handle special cases of types.
    if is_built_in_or_extension_type(orig_func):
        return False
    if is_class(orig_func):
        if hasattr(orig_func, '__slots__'):
            return False
        wrapped_func = dump_signature_of_class(orig_func, func_name_str, output_file=output_file)
        setattr(module_obj, func_name, wrapped_func)
        return True
    else:
        if is_callable(orig_func):
            wrapped_func = dump_signature_of_function(orig_func, func_name_str, output_file=output_file)
            setattr(module_obj, func_name, wrapped_func)
            return True
        else:
            return False


def hijack(output_file):
    api_file = __file__.replace("hijack.py", "operator_list.txt")
    with open(api_file, 'r') as fr:
        apis = fr.readlines()
    print('Number of total instrumented operator API: ', len(apis))
    for i, api in enumerate(apis):
        api = api.strip()
        hijack_api(tf, api[3:], output_file)
