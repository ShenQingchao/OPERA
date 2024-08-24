import os
import sys
import subprocess
import multiprocessing
import datetime
import shutil

import warnings
warnings.filterwarnings('ignore')


def run_subprocess(python_program_gpu):
    python_program, gpu_id = python_program_gpu
    this_start_time = datetime.datetime.now()
    print(f"Execute subprocess: {python_program}")
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        run_flag = subprocess.run([f'python', python_program], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=80)
        if run_flag.returncode == 0:  # run well
            output = run_flag.stdout.decode('utf-8')
            output_final = ''
            for line in output.split("\\n"):
                output_final += line
            print(output_final)
        else:  # invalid test cases
            err_output = run_flag.stderr.decode('utf-8')
            output_final = ''
            for line in err_output.split("\\n"):
                output_final += line

            print(f">>>> [Warning] Check the test case in file {python_program}")
            print(output_final)
    except subprocess.TimeoutExpired:
        print(f"[Timeout] for {python_program}, continue...")

    # record the consumed time
    tc_id = python_program.split("test_")[-1][:-3]
    time_consume = (datetime.datetime.now() - this_start_time).seconds
    with open(f"time_record_{SUT}_{frame}.txt", 'a')as f:
        f.write(f"{tc_id}\t{time_consume}\n")
    return


def gen_test_case_file(SUT, test_str, frame, save_dir, test_id):
    keras_run_head = f"""from test_{SUT}_keras import layer_test
from tensorflow import keras
import tensorflow as tf
import numpy as np

"""
    torch_run_head = f"""import torch
from torch.nn import Module
from test_{SUT}_torch import verify_model
from numpy import inf
def tensor(x):
    return x

"""
    onnx_run_head = f"""import onnx
from test_{SUT}_onnx import make_graph

"""

    if frame == 'keras':
        test_str = keras_run_head + test_str
    elif frame == 'torch':
        test_str = torch_run_head + test_str
    elif frame == 'onnx':
        test_str = onnx_run_head + test_str
    else:
        assert False, f"Unsupported frontends {frame} yet!"

    save_dir = f"{save_dir}_{frame}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(f"test_{SUT}_{frame}.py", save_dir)
    save_test_file_path = os.path.join(save_dir, f'test_{test_id}.py')
    if not os.path.exists(save_test_file_path):
        with open(save_test_file_path, 'w', encoding='utf-8') as test_f:
            test_f.write(test_str)
    return save_test_file_path


def _load_test_from_file(test_file, cnt=1):
    # support keras, pytorch and ONNX!
    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    all_test_str_list = []

    this_test_case = ''
    for line in all_lines:
        if line.startswith("verify_model") or line.startswith("layer_test") or line.startswith("make_graph"):
            line = line.strip()[:-1]
            if line.endswith(','):
                line += f"count={cnt},)"
            else:
                line += f",count={cnt},)"
            this_test_case += line
            if this_test_case not in all_test_str_list:  # deduplicate
                all_test_str_list.append(this_test_case)
                cnt += 1
            this_test_case = ''
        else:
            this_test_case += line
    print(f"Number of collected tests in {frame} is : {len(all_test_str_list)}")
    return all_test_str_list


def run_all_test(test_file, SUT, frame, begin_id=1):
    test_dir = f"all_{frame}_tc"
    all_test_files = []

    if not os.path.exists(test_dir):
        all_test_str_list = _load_test_from_file(test_file, begin_id)
        for cnt, test_str in enumerate(all_test_str_list):
            test_str = test_str.strip()
            save_test_file_path = gen_test_case_file(SUT, test_str, frame, test_dir, cnt+begin_id)
            all_test_files.append(save_test_file_path)
    else:
        for tc in os.listdir(test_dir):
            if f'test_{SUT}_{frame}.py' in tc:
                continue
            elif tc.endswith('.py'):
                tc_file = os.path.join(test_dir, tc)
                all_test_files.append(tc_file)

    # Execute the tests
    # for tc in all_test_files:
    #     run_subprocess(tc)

    gpu_num = 8
    all_test_files_gpu = []
    for i, file in enumerate(all_test_files):
        all_test_files_gpu.append((file, i%gpu_num))

    pool = multiprocessing.Pool(processes=2*gpu_num)
    pool.map(run_subprocess, all_test_files_gpu)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # cd keras/fuzz
    # python run_fuzz.py ../data/demo.py openvino keras
    # python run_fuzz.py ../data/demo.py openvino onnx
    # python run_fuzz.py ../data/combined_sources_keras_test_41986.py tvm keras

    starttime = datetime.datetime.now()
    collected_test_cases_file = sys.argv[1]
    SUT = sys.argv[2]    # [tvm, openvino]
    frame = sys.argv[3]  # [keras, torch, onnx]
    # run_all_test(collected_test_cases_file, SUT, frame)
    test_begin_id = 1
    if 'deeprel' in collected_test_cases_file:
        if frame == 'torch':
            test_begin_id = 64757
        elif frame == 'keras':
            test_begin_id = 41985
    run_all_test(collected_test_cases_file, SUT, frame, test_begin_id)
    endtime = datetime.datetime.now()
    print("Finish all, time consuming(min): ", (endtime - starttime).seconds/60)
