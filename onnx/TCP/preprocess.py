all_test_cases = {}
count = 0

def process_tc_from_file(tc_file_name):
    # Skip the operators of aionnx_preview_training
    skip_list = ['LinearRegressor', 'Adagrad', 'Adam', 'ArrayFeatureExtractor', 'Binarizer', 'Momentum']
    with open(tc_file_name, 'r', encoding='utf-8') as intput_f:
        all_lines = intput_f.readlines()

    this_test_case = ''
    global all_test_cases, count
    for i, line in enumerate(all_lines):
        if line.startswith("make_graph") and (not this_test_case):
            if "UNDEFINED" in line:
                continue
            else:
                op_type = line.split("op_type=")[1].split(", ")[0]
                if op_type in skip_list:
                    continue
                else:
                    this_test_case += line
                    if line.endswith(")\n"):
                        count += 1
                        all_test_cases[count] = this_test_case
                        this_test_case = ''
        else:
            if this_test_case:
                if "UNDEFINED" in line:
                    this_test_case = ''
                    continue
                this_test_case += line
                if line.endswith(")\n"):
                    count += 1
                    all_test_cases[count] = this_test_case
                    this_test_case = ''
    return all_test_cases


if __name__ == '__main__':
    origin_test_file = "../data/original_migrated_onnx_tc.py"
    # origin_test_file = "/share_host/TVMFT/BorrowTests/keras/all_borrow_test.py"
    mitigated_tc_dict = process_tc_from_file(origin_test_file)
    print(count)

    unique_items = set(mitigated_tc_dict.items())
    with open('output.txt', 'w') as f:
        for key, value in unique_items:
            print(key, value, file=f)
    # tvm_equipped_test_file = "data/tvm_onnx_all_test.py"
    # tvm_tc_dict = load_tc_from_file(tvm_equipped_test_file).all_tc
    #
    # save_test_file = "ranked_test_case.py"
    # run_tcp(mitigated_tc_dict, tvm_tc_dict, max_instance_number=100, save_file=save_test_file)

