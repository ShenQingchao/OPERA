import os
import datetime
import multiprocessing
import subprocess

head_import_codes = """
import torch
from torch.nn import Module
from test_tvm import verify_model
from numpy import inf
import datetime
starttime = datetime.datetime.now()

def tensor(x):
    return x

"""

tail_codes = """

endtime = datetime.datetime.now()
print("Finish all, time consuming(min): ", (endtime - starttime).seconds/60)
"""


def collect_all_test_files(test_dir):
    py_files = []
    for dir_path, dir_names, filenames in os.walk(test_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                py_files.append(os.path.join(dir_path, filename))
        return py_files


def post_process(collected_test_file, new_file_name):
    """
    1. correct the test_case id number
    2. split the file into some sub files which each file containing 10k test cases
    3. add import code to each test file.
    :param collected_test_file:
    :param new_file_name:
    :return:
    """
    new_count = 0
    file_id = 0

    with open(collected_test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    with open(f"{new_file_name}_0.py", 'a', encoding='utf-8') as f:
        f.write(head_import_codes)

    for line in all_lines:
        new_file_path = f"{new_file_name}_{file_id}.py"
        with open(new_file_path, 'a', encoding='utf-8') as n_f:
            if line.startswith("# test_id:"):
                n_f.write(f"# test_id: {new_count}\n")
                new_count += 1
                if new_count % 1000000000 == 0:
                    with open(f"{new_file_name}_{file_id}.py", 'a', encoding='utf-8') as next_f:
                        next_f.write(tail_codes)
                    file_id += 1

                    with open(f"{new_file_name}_{file_id}.py", 'a', encoding='utf-8') as next_f:
                        next_f.write(head_import_codes)

            else:
                n_f.write(line)

    return new_count


def exec_all_test_files(path_dir="./", test_file_name="borrow_all_new"):
    starttime = datetime.datetime.now()
    all_collected_test_files = []
    for file_name in os.listdir(path_dir):
        if file_name.startswith(test_file_name+"_"):
            all_collected_test_files.append(file_name)
            # os.system(f"python {file_name} &> {test_file_name}.log")

    with multiprocessing.Pool(8) as pool:
        pool.map(exec_python_file,  all_collected_test_files)

    endtime = datetime.datetime.now()
    time_consume = (endtime - starttime).seconds / 60
    print(f'time_consume:{time_consume} min')
    return time_consume


def exec_python_file(file_name):
    subprocess.call(['python', file_name])
    print(f"execute file: {file_name}")
    os.system(f"python {file_name} &> log.{file_name[:-3]}")   # 阻塞进程


if __name__ == '__main__':
    test_dir = "torch_unit_test"

    """
    all_test_files = collect_all_test_files(test_dir)
    # print(all_test_files)
    pool = multiprocessing.Pool(processes=20)
    pool.map(exec_python_file(), all_test_files)
    pool.close()
    pool.join()
    """

    # rename the count_id, spilt test_files and add import and time collection code.
    # all_test_num = post_process("borrow_all.py", "borrow_all_new")
    # print("all_test_num", all_test_num)

    # exec all test files and collect the times
    time_consuming = exec_all_test_files()
    print("Total executing time:", time_consuming)
