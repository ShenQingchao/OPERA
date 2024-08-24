import sys

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor
import torch
import traceback
import re

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
    file_name = file_name[len("/workplace/software/tvm/tvm_/"):]
    exc_type = type(e).__name__
    print(str(e))
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

    with open("../data/detected_bugs_tvm.txt", 'a', encoding='utf-8') as f:
        f.write(bug_info_str)


def verify_model(
        model_name,
        input_data=None,
        custom_convert_map=None,
        rtol=1e-5,
        atol=1e-5,
        expected_ops=None,
        kind="graph",
        check_correctness=True,
        cpu_only=True,
        count=0,
):
    try:
        """Assert that the output of a compiled model matches with that of its
        baseline."""
        input_data = [] if input_data is None else input_data
        if len(input_data[0].size()) == 0:  # input_shape is empty skip it.
            return
        custom_convert_map = custom_convert_map or {}
        expected_ops = expected_ops or []
        # if isinstance(model_name, str):
        #     baseline_model, baseline_input = load_model(model_name)
        if isinstance(input_data, list):
            baseline_model = model_name
            baseline_input = input_data
        elif isinstance(input_data, torch.Tensor) or not input_data.shape:
            baseline_model = model_name
            # print(baseline_model)
            baseline_input = [input_data]
        else:
            assert False, "Unexpected input format"
        if torch.cuda.is_available():
            if isinstance(baseline_model, torch.nn.Module):
                baseline_model = baseline_model.cuda()
            baseline_input = [inp.cuda() for inp in baseline_input]

        with torch.no_grad():
            baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])
            # print(baseline_input)

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

        input_names = [f"input{idx}" for idx, _ in enumerate(baseline_input)]
        input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    except Exception as e:
        # print(f"[test-{count}] torch error: ", e)
        return  # TODO: modify the test_case extraction method to get correct api_call rather than ignore it.

    # load to tvm and compile
    try:
        mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
        # print(mod)
        for arg in mod["main"].params[: len(input_names)]:
            assert arg.name_hint in input_names
        # print('input_names', input_names)
        compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input]))

        targets = ["llvm"]
        if not cpu_only:
            targets.append("cuda")
    except Exception as e:
        if 'operators are not implemented' in str(e) or '(0 <= i && i < p->size_) is false: IndexError:' in str(
                e) or 'unable to show the following types match' in str(
                e):  # skip the common error to avoid numerous crash message
            return
        crash_message = extract_crash_message(e)
        record_bug(count, 'crash', type(model_name).__name__, crash_message=crash_message)
        return
    try:
        with tvm.transform.PassContext(opt_level=3):
            for target in targets:
                if not tvm.runtime.enabled(target):
                    continue
                dev = tvm.device(target, 0)
                kind_list = ['graph', ]  # 'vm', 'debug']
                for kind in kind_list:
                    try:
                        exe = relay.create_executor(
                            kind, mod=mod, params=params, device=dev, target=target
                        ).evaluate()
                        # print(dev, target)
                        result = exe(**compiled_input)
                        if not isinstance(result, list):
                            result = [result]
                    except Exception as e:
                        crash_message = extract_crash_message(e)
                        record_bug(count, 'crash', type(model_name).__name__, crash_message=crash_message)
                        return

                    try:
                        for i, baseline_output in enumerate(baseline_outputs):
                            output = result[i].numpy()
                            assert_shapes_match(baseline_output, output)
                            if check_correctness:
                                tvm.testing.assert_allclose(baseline_output, output, rtol=rtol, atol=atol)
                    except AssertionError as e:
                        record_bug(count, 'wrong results', type(model_name).__name__, 'wrong result')
                        return

        del model_name
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f'[test-{count}:{str(model_name)}] tvm compile error:', e)
        crash_message = extract_crash_message(e)
        record_bug(count, 'crash', type(model_name).__name__, crash_message=crash_message)
