import sys

import torch
from torch.nn import Module
from torch2trt import torch2trt

import traceback
import re
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

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
    exc_type = type(e).__name__
    stack_trace = str(e).strip().split("\n")[-1]
    if stack_trace.endswith(':'):
        stack_trace = stack_trace[:-1]
    stack_trace = stack_trace.split(':')[-1].strip()
    pattern = r"[\[\(].*?[\]\)]"
    stack_trace = re.sub(pattern, "", stack_trace)
    print(f">>>>>>>>>>>>>>>>>>>Bug Info: {stack_trace}")

    crash_message = f"{exc_type}_{file_name}_{line_number}_{stack_trace}"
    return crash_message


def record_bug(bug_id, bug_type, op, crash_message=''):
    bug_info_str = f"{bug_id}\t{bug_type}\t{op}\t{crash_message}\n"

    with open("../data/detected_bugs_trt.txt", 'a', encoding='utf-8') as f:
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
        elif isinstance(input_data, torch.Tensor):
            baseline_model = model_name
            baseline_input = [input_data]
        else:
            assert False, "Unexpected input format"

        with torch.no_grad():
            baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])
            baseline_outputs = baseline_outputs.cpu().numpy()
        # input_shapes = list([inp.shape for inp in baseline_input])
    except Exception as e:
        print(f"[test-{count}] torch error: ", e)
        return  # TODO: modify the test_case extraction method to get correct api_call rather than ignore it.
    # res_dlc = compile_torch(count, trace, input_shapes, baseline_input)
    try:
        model_trt = torch2trt(model_name.cuda(), baseline_input)
        res_dlc = model_trt(*baseline_input).cpu().numpy()
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
        np.testing.assert_allclose(baseline_outputs, res_dlc, rtol=rtol, atol=atol)
        # for i, baseline_output in enumerate(baseline_outputs):
        #    output = res_dlc[i]
        #    # print(output.shape)
        #    assert_shapes_match(baseline_output, output)
        #    if check_correctness:
        #        np.testing.assert_allclose(baseline_output, output, rtol=rtol, atol=atol)
    except AssertionError as e:
        print(e)
        record_bug(count, 'wrong results', type(model_name).__name__, 'wrong result')
        return
    print("[success] This test case passed!")


if __name__ == '__main__':
    # class pad(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.pad(args[0], (25, 25), )
    # para_0 = torch.randn([1, 6, 51], dtype=torch.complex64)
    # verify_model(pad().float().eval(), input_data=para_0)

    # input_data = [torch.randn([0, 3, 3, 4, 5], dtype=torch.float32)]
    # print(input_data)
    # verify_model(torch.nn.Conv3d(3, 4, (2, 3, 4), ).eval(),
    #              input_data=input_data)

    # test_id: 4134
    # para_0 = torch.randn([1, 16, 4, 4], dtype=torch.float32)
    #
    # class avg_pool2d(Module):
    #     def forward(self, input):
    #         return torch.nn.functional.avg_pool2d(input, ceil_mode=True,  kernel_size=(1, 2), padding=(0, 1), stride=2)
    # verify_model(avg_pool2d().float().eval(), input_data=para_0)

    # test_id: 13877
    # para_0 = torch.randn([1, 9216], dtype=torch.float32)
    # class dropout(Module):
    #     def forward(self, *args):
    #         return torch.nn.functional.dropout(args[0])
    # verify_model(dropout().float().eval(), input_data=para_0)

    para_0 = torch.randn([6, 176, 9, 9], dtype=torch.float32).cuda()


    class max_pool2d(Module):
        def forward(self, *args):
            return torch.nn.functional.max_pool2d(args[0], kernel_size=3, stride=1, padding=0, dilation=2,
                                                  ceil_mode=True, )


    verify_model(max_pool2d().float().eval(), input_data=para_0)
