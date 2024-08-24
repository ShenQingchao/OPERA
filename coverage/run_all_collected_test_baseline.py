import csv
import os
import shutil
import sys
import subprocess
import multiprocessing
import datetime
import json
import warnings

warnings.filterwarnings('ignore')

dlc = 'tvm'  # TODO: change it when modify dlc

keras_run_head = f"""from test_{dlc}_keras import layer_test
import keras
import tensorflow as tf
import numpy as np

"""

torch_run_head = f"""import torch
from torch.nn import Module
from test_{dlc}_torch import verify_model
from numpy import inf
def tensor(x):
    return x

"""

onnx_run_head = f"""import onnx
from test_{dlc}_onnx import make_graph
"""

comb_line = {}
comb_branch = {}
program_to_str = {}
covered_lines = {}
covered_branches = {}
delta_lines = {}
delta_branches = {}
valid_test = []


def run_subprocess(python_program):
    print(f"Execute subprocess: {python_program}")
    testcase = os.path.basename(python_program).replace('.py', '').replace('test_', '')
    rc_file = '.coveragerc'
    cov_file = os.path.join(f'{frame}_tests', '.coverage.' + testcase)
    json_file = python_program.replace('.py', '.json')
    if os.path.exists(json_file):  # skip the executed tests
        print('skip this executed test!')
        return python_program
    print(f"execute {python_program}, cov_file {cov_file}, json_file {json_file} ['coverage', 'run', '--data-file={cov_file}', '--rcfile={rc_file}', python_program]")
    try:
        run_flag = subprocess.run(['coverage', 'run', f'--data-file={cov_file}', f'--rcfile={rc_file}', python_program],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40)
    except subprocess.TimeoutExpired:
        print('timeout for 40s')
        return None
    except Exception as e:
        print(e)

    if run_flag.returncode == 0:  # run well
        output = run_flag.stdout.decode('utf-8')
        output_final = ''
        for line in output.split("\n"):
            output_final += line
        print(output_final)
        # print('begin coverage json')
        subprocess.run(['coverage', 'json', f'--data-file={cov_file}', '-o', f'{json_file}'])
        # report = subprocess.run(['coverage', 'report', f'--data-file={cov_file}', f'--rcfile={rc_file}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # stdout = report.stdout
        # stderr = report.stderr
        # print(python_program, stdout, stderr)
        # 解析JSON数据

        # last_line = report.stdout.decode('utf-8').split('\r\n')[-2]
        # print(f'last_line {last_line}')
        # return None

        # is_valid[python_program] = True
        # valid_test.append(python_program)
        # print(f'coverage {python_program}: [line: {cov_line}, branch: {cov_branch}]')
        # os.remove(json_file)
        # print(f'{python_program} is appended')
        return python_program
    else:
        err_output = run_flag.stderr.decode('utf-8')
        output_final = ''
        for line in err_output.split("\\n"):
            output_final += line

        print(f">>>> [Warning] Check the test case in file {python_program}")
        print(output_final)
        print(run_flag.stdout.decode('utf-8'))
    return None


def get_delta2(programs):
    line = 0
    with open('cov.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for program in programs:
            json_file = program.replace('.py', '.json')
            print(f'delta {program}')
            if not os.path.exists(json_file):
                covered_lines[program] = 0
                delta_lines[program] = 0
            else:
                with open(json_file) as f:
                    data = json.load(f)
                    # print(f"data is {data['totals']}")
                    cov_line = data['totals']['covered_lines']
                    covered_lines[program] = cov_line

                    cov_line = 0
                    file_data = data['files']
                    for file in file_data.keys():
                        if file not in comb_line:
                            comb_line[file] = set()
                            comb_branch[file] = set()
                        lines = file_data[file]['executed_lines']
                        for l in lines:
                            comb_line[file].add(l)
                        cov_line += len(comb_line[file])
                    print(f'cov_line: {cov_line}')
                    delta_lines[program] = cov_line - line
                    line = cov_line
            writer.writerow([program, covered_lines[program], delta_lines[program],])
    return


def line_rank(test_files):
    sorted_files = sorted(test_files, key=lambda t: -covered_lines[t])
    with open(f"{frame}_{dlc}_line.py", 'w') as out:
        for file in sorted_files:
            out.write(program_to_str[file] + '\n')
    return


def delta_line_rank(test_files):
    sorted_files = sorted(test_files, key=lambda t: -delta_lines[t])
    with open(f"{frame}_{dlc}_delta_line.py", 'w') as out:
        for file in sorted_files:
            out.write(program_to_str[file] + '\n')
    pass


def gen_test_case_file(test_str, frame, test_id):
    save_dir = f"{frame}_tests"
    save_test_file_path = os.path.join(save_dir, f'test_{test_id}.py')
    program_to_str[save_test_file_path] = test_str

    if os.path.exists(save_test_file_path):
        return save_test_file_path
    if frame == 'keras':
        test_str = keras_run_head + test_str
    elif frame == 'torch':
        test_str = torch_run_head + test_str
    elif frame == 'onnx':
        test_str = onnx_run_head + test_str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy2(f'test_{dlc}_{frame}.py', f'{save_dir}')
    # os.system(f"cp test_{dlc}_{frame}.py {save_dir}")
    with open(save_test_file_path, 'w', encoding='utf-8') as test_f:
        test_f.write(test_str)

    return save_test_file_path


def load_test_from_file(test_file):
    # support keras and pytorch now!
    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    all_test_str_list = []

    this_test_case = ''
    cnt = 0   # todo: fix this bug later with 1
    for line in all_lines:
        if line.startswith("verify_model") or line.startswith("layer_test") or line.startswith("make_graph"):
            line = line.strip()[:-1]
            if line.endswith(','):
                line += f"count={cnt},)"
            else:
                line += f",count={cnt},)"
            this_test_case += line
            if this_test_case not in all_test_str_list:
                cnt += 1
                # print(cnt)
                all_test_str_list.append(this_test_case)
            this_test_case = ''
        else:
            this_test_case += line
    return all_test_str_list


def run_all_test(test_file, frame='keras'):
    all_test_str_list = load_test_from_file(test_file)
    print(f"The collected test cases number in {frame} is : {len(all_test_str_list)}")
    all_test_files = []
    for cnt, test_str in enumerate(all_test_str_list):
        test_str = test_str.strip()
        save_test_file_path = gen_test_case_file(test_str, frame, cnt+1)
        all_test_files.append(save_test_file_path)
    shutil.copy(f'test_{dlc}_{frame}.py', f'{frame}_tests')

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2)
    results = pool.map(run_subprocess, all_test_files)
    pool.close()
    pool.join()

    get_delta2(all_test_files)
    line_rank(all_test_files)
    delta_line_rank(all_test_files)


if __name__ == '__main__':
    # !!!!!!!!!! change the dlc in the header !!!!!!!!!!!!!!!!!
    starttime = datetime.datetime.now()
    frame = sys.argv[1]  # torch or keras or onnx
    collected_test_cases_file = sys.argv[2]  # test_file_path
    run_all_test(collected_test_cases_file, frame)
    endtime = datetime.datetime.now()
    print("Finish all, time consuming(min): ", (endtime - starttime).seconds / 60)
