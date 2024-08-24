import re
import random


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
        return f"torch.randn({input_shape}, dtype={input_dtype})"


def correct_tc(origin_tc):
    pattern = re.compile(r"input=\{.*\},")
    new_tc = re.sub(pattern, "", origin_tc)
    if origin_tc != new_tc:
        input_tensor = re.findall(pattern, origin_tc)[0]
        input_tensor = input_tensor[len('input='):]
        print(input_tensor)
        input_data = input_dict2data(eval(input_tensor[:-1]))
        input_data = "para_0 = " + input_data + '\n'
        print(input_data)
        new_tc = new_tc.replace('\nclass ', f'\n{input_data}class ')
    return new_tc


if __name__ == '__main__':
    origin_tc = '''class dropout3d(Module):
            def forward(self, *args):
                return torch.nn.functional.dropout3d(args[0], input={'shape': [20, 1, 4], 'dtype': 'torch.float64'},)
        verify_model(dropout3d().float().eval(), input_data=para_0)
        '''
    new_tc = correct_tc(origin_tc)
    print(len(new_tc))

    with open('_combined_source_torch_test_64756_old.py', 'r') as f:
        all_lines = f.read()

    print(type(all_lines))
    # split_pattern = "# test_id: .*?\n"
    split_pattern ="\n\n\n"
    all_tc = re.split(split_pattern, all_lines)
    for tc in all_tc:
        new_tc = correct_tc(tc)
        with open('original_migrated_torch_tc.py', 'a') as w:
            w.write(new_tc+"\n\n\n")
