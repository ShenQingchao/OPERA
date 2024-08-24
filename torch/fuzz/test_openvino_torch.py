import sys

import openvino as ov
import torch
from torch.nn import Module

import traceback
import re
import numpy as np
import os


sys.setrecursionlimit(10000)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def assert_shapes_match(tru, est):
    """Verfiy whether the shapes are equal"""
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


def extract_crash_message(e):
    tb = traceback.extract_tb(e.__traceback__)
    file_name, line_number, _, _ = tb[-1]
    file_name = file_name.split("site-packages")[-1]
    exc_type = type(e).__name__
    stack_trace = str(e).split("Summary:")[0].strip().split("\n")[-1]
    # if stack_trace.endswith(':'):
    #     stack_trace = stack_trace[:-1]
    # stack_trace = stack_trace.split(':')[-1].strip()
    pattern = r"[\[\(].*?[\]\)]"
    stack_trace = re.sub(pattern, "", stack_trace)
    print(f">>>>>>>>>>>>>>>>>>>Bug Info: {stack_trace}")
    crash_message = f"{exc_type}_{file_name}_{line_number}_{stack_trace}"
    return crash_message


def record_bug(bug_id, bug_type, op, crash_message=''):
    bug_info_str = f"{bug_id}\t{bug_type}\t{op}\t{crash_message}\n"

    with open("../data/detected_bugs_ov.txt", 'a', encoding='utf-8') as f:
        f.write(bug_info_str)


def verify_model(
        model_name,
        input_data=None,
        rtol=1e-3,
        atol=1e-3,
        check_correctness=True,
        count=0,
):
    try:
        """Assert that the output of a compiled model matches with that of its
        baseline."""
        input_data = [] if input_data is None else input_data
        if len(input_data[0].size()) == 0:  # input_shape is empty and skip it.
            print("[Warning] skip the test case due to the empty inputs")
            return
        if isinstance(input_data, list):
            baseline_model = model_name
            baseline_input = input_data
        elif isinstance(input_data, torch.Tensor) or not input_data.shape:
            baseline_model = model_name
            # print(baseline_model)
            baseline_input = [input_data]
        else:
            assert False, "Unexpected input format"
        # if torch.cuda.is_available():
        #     if isinstance(baseline_model, torch.nn.Module):
        #         baseline_model = baseline_model.cuda()
        #     baseline_input = [inp.cuda() for inp in baseline_input]

        with torch.no_grad():
            baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

        if isinstance(baseline_outputs, tuple):
            for out in baseline_outputs:
                print(type(out))
            baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
        else:
            baseline_outputs = (baseline_outputs.cpu().numpy(),)

        trace = torch.jit.trace(baseline_model, [input.clone() for input in baseline_input])
        if isinstance(baseline_model, torch.nn.Module):
            trace = trace.float().eval()

            if torch.cuda.is_available():
                trace = trace.cuda()
            else:
                trace = trace.cpu()

        # input_names = [f"input{idx}" for idx, _ in enumerate(baseline_input)]
        # input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
        input_shapes = list([inp.shape for inp in baseline_input])
        trace = torch.jit.freeze(trace)
        # print(input_shapes)
    except Exception as e:
        print(f"[test-{count}] torch error: ", e)
        return  # TODO: modify the test_case extraction method to get correct api_call rather than ignore it.
    # res_dlc = compile_torch(count, trace, input_shapes, baseline_input)
    try:
        res_dlc = compile_torch(count, trace, input_shapes, baseline_input)
    except Exception as e:
        if 'support' in str(e) or 'not allowed' in str(e) or "No conversion rule" in str(e) or 'type must be' in str(e):
            print(e)
            print("[Warning] trigger an unsupported behavior")
        else:
            print(f'[Bug in DLC] using test: {type(model_name).__name__}; id= {count}')
            print(e)
            crash_message = extract_crash_message(e)
            record_bug(count, 'crash', type(model_name).__name__, crash_message=crash_message)
        return
    try:
        # print(len(baseline_outputs))
        for i, baseline_output in enumerate(baseline_outputs):
            output = res_dlc[i]
            # print(output.shape)
            assert_shapes_match(baseline_output, output)
            if check_correctness:
                np.testing.assert_allclose(baseline_output, output, rtol=rtol, atol=atol)
    except AssertionError as e:
        print(e)
        record_bug(count, 'wrong results', type(model_name).__name__, 'wrong result')
        return
    print("[success] This test case passed!")


def compile_torch(cnt, model, input_shapes, input_data):
    # [reference](https://docs.openvino.ai/2023.1/openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch.html)
    temp_model_dir = "_temp_model"
    if not os.path.exists(temp_model_dir):
        os.mkdir(temp_model_dir)
    # print(input_shapes)
    ov_model = ov.convert_model(model, example_input=input_data)  # input_shape only get the shape of first element of list
    print("convert to ov successfully...")
    ir_path = f"{temp_model_dir}/_temp_OVIR_{cnt}.xml"
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    core = ov.Core()
    model = core.read_model(ir_path)

    compiled_model = core.compile_model(model=model, device_name="CPU")  # CPU,GPU,AUTO

    # show the model structure
    # input_key = compiled_model.input(0)
    output_key = compiled_model.outputs
    # network_input_shape = input_key.shape

    # show the model structure
    # input_key = compiled_model.input(0)
    output_key = compiled_model.outputs
    # print("output_key:", output_key)
    result = []
    for output in output_key:
        result.append(compiled_model(input_data)[output])
    return result


if __name__ == '__main__':
    # para_0 = torch.randn([5, 8, 6], dtype=torch.float32)
    # class max_pool2d(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.max_pool2d(args[0], kernel_size=3, stride=1, padding=0, dilation=2,
    #                                               ceil_mode=False, )
    # verify_model(max_pool2d().float().eval(), input_data=para_0)

    # para_0 = torch.randn([1, 5, 6, 7], dtype=torch.float32)
    # para_1 = (3, 6, 5)
    # class avg_pool3d(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.avg_pool3d(args[0], para_1, )
    # verify_model(avg_pool3d().float().eval(), input_data=para_0)

    # test_id: 38979
    # para_0 = torch.randint(1, 100, [1, 3, 7, 6], dtype=torch.int64)
    # para_1 = torch.randint(1, 100, [3, 4, 3, 3], dtype=torch.int64)
    # para_2 = torch.randint(1, 100, [4], dtype=torch.int64)
    #
    # class conv_transpose2d(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.conv_transpose2d(args[0], para_1, para_2,)
    # verify_model(conv_transpose2d().float().eval(), input_data=para_0)

    # para_0 = torch.randint(1, 100, [1, 2, 4, 5], dtype=torch.int64)
    # para_1 = torch.randint(1, 100, [2, 2, 2, 3], dtype=torch.int64)
    # para_2 = torch.randint(1, 100, [4], dtype=torch.int64)
    # para_3 = (1, 1)
    # para_4 = (0, 0)
    # para_5 = (0, 0)
    # para_6 = 2
    # para_7 = (1, 1)
    #
    # class conv_transpose2d(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.conv_transpose2d(args[0], para_1, para_2, para_3, para_4, para_5, para_6,
    #                                                     para_7, )
    # verify_model(conv_transpose2d().float().eval(), input_data=para_0)

    # test_id: 8204
    para_0 = torch.randn([1, 5, 6, 7], dtype=torch.float64)
    para_1 = (5, 2, 5)

    class avg_pool3d(Module):
        def forward(self, *args):
            return torch.nn.functional.avg_pool3d(args[0], para_1, )

    verify_model(avg_pool3d().float().eval(), input_data=para_0)
