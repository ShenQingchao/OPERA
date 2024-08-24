import heapq
import random
import time
import os
from TCP.case import TC, TCDict


def run_tcp(tc_dict, tvm_equipped_tc_dict, max_instance_number=1, save_file='ranked_test_case.py'):
    tc_dict.rank_layer(tvm_equipped_tc_dict=tvm_equipped_tc_dict)
    tvm_dict = TCDict()
    all_dict = TCDict()
    rate = {}

    for layer, tc_list in tvm_equipped_tc_dict.items():
        if layer not in tc_dict.all_tc:
            continue
        for tc in tc_list:
            tc_dict.add2selected_tc_dict(tc)
            all_dict.add(tc)
            all_dict.add2selected_tc_dict(tc)
            tvm_dict.add(tc)
            tvm_dict.add2selected_tc_dict(tc)
    print('layers', len(tc_dict.all_tc.keys()))
    for layer, tc_list in tc_dict.all_tc.items():
        r = 1.0
        for tc in tc_list:
            all_dict.add(tc)
            all_dict.add2selected_tc_dict(tc)
        if layer in tvm_dict.all_tc.keys():
            tvm_tc_dict = tvm_dict.all_selected_tc[layer]
            all_tc_dict = all_dict.all_selected_tc[layer]
            for key, val in all_tc_dict.items():
                if key in tvm_tc_dict:
                    r *= 1.0 * len(tvm_tc_dict[key]) / len(all_tc_dict[key])
                else:
                    r *= 1.0 / len(all_tc_dict[key])
            tvm_tc_dict_pair = tvm_dict.all_selected_tc_pair[layer]
            all_tc_dict_pair = all_dict.all_selected_tc_pair[layer]
            for key, val in all_tc_dict_pair.items():
                if key in tvm_tc_dict_pair:
                    r *= 1.0 * len(tvm_tc_dict_pair[key]) / len(all_tc_dict_pair[key])
                else:
                    r *= 1.0 / len(all_tc_dict_pair[key])
        else:
            r = 0
        r = 1.0 - r
        rate[layer] = r
        print(f'{layer} : {r}')

    heap = []
    tests = []
    length = 0
    for layer_name, instance_list in tc_dict.all_tc.items():
        if len(instance_list) == 0:  # skip it if the layer group is empty
            continue
        length += len(tc_dict.all_tc[layer_name])
        this_selected_tc, max_distance = tc_dict.select_instance(layer_name)
        if max_distance == 0:
            continue
        heapq.heappush(heap, (-rate[layer_name] * max_distance, this_selected_tc))
    will_delete_layer_group = []

    while len(heap) != 0:

        max_distance, this_selected_tc = heapq.heappop(heap)
        with open(save_file, 'a', encoding='utf-8') as out_f:
            line_cnt = this_selected_tc.id
            out_f.write(this_selected_tc.test_cmd_str.strip()[:-1] + f",count={line_cnt},)\n")
        tc_dict.add2selected_tc_dict(this_selected_tc)
        tc_dict.remove(this_selected_tc)
        layer_name = this_selected_tc.graph
        if layer_name == 'SequenceConstruct':
            print(f'DEBUG {max_distance} {this_selected_tc.abstract_tc} -> {tc_dict.all_selected_tc[layer_name]}')
        if max_distance == 0 or len(tc_dict.all_tc[layer_name]) == 0:
            # tests.extend(tc_dict.all_tc[layer_name])
            # del tc_dict.all_tc[layer_name]
            continue
        else:
            selected_tc, distance = tc_dict.select_instance(layer_name)
            heapq.heappush(heap, (-rate[layer_name] * distance, selected_tc))

    for layer_name, instance_list in tc_dict.all_tc.items():
        tests.extend(instance_list)
    random.seed(1)
    random.shuffle(tests)

    print(f'len of tests {len(tests)}')
    with open(save_file, 'a', encoding='utf-8') as out_f:
        for test in tests:
            line_cnt = test.id
            out_f.write(test.test_cmd_str.strip()[:-1] + f', count={line_cnt},)\n')


def preprocess_onnx_tc_from_file(tc_file_name):
    all_test_cases = {}
    count = 0
    # Skip the operators of aionnx_preview_training
    skip_list = ['LinearRegressor', 'Adagrad', 'Adam', 'ArrayFeatureExtractor', 'Binarizer', 'Momentum']
    # skip_list = []
    with open(tc_file_name, 'r', encoding='utf-8') as intput_f:
        all_lines = intput_f.readlines()

    this_test_case = ''
    for i, line in enumerate(all_lines):
        if line.startswith("make_graph") and (not this_test_case):
            op_type = line.split("op_type='")[1].split("', ")[0]
            if op_type in skip_list:
                continue
            else:
                this_test_case += line
                if line.endswith(")\n"):
                    # count += 1
                    all_test_cases[i] = this_test_case
                    this_test_case = ''
        else:
            if this_test_case:
                this_test_case += line
                if line.endswith(")\n"):
                    count += 1
                    all_test_cases[count] = this_test_case
                    this_test_case = ''
    return all_test_cases


def load_tc_from_file(tc_file_name):
    all_test_cases = preprocess_onnx_tc_from_file(tc_file_name)
    tc_dict = TCDict()
    for key, value in all_test_cases.items():
        new_tc = TC(key, value)
        if new_tc.is_valid:
            tc_dict.add(new_tc)
    return tc_dict


if __name__ == '__main__':
    front = 'onnx'
    SUT = "trt"  # ov, tvm, trt
    origin_test_file = f"../data/original_migrated_{front}_tc.py"
    SUT_equipped_test_file = f"../data/{SUT}_equipped_{front}_tc.py"
    save_test_file = f"../data/ranked_{front}_tc_4_{SUT}.py"
    if os.path.exists(save_test_file):
        os.remove(save_test_file)

    start = time.time()
    mitigated_tc_dict = load_tc_from_file(origin_test_file)
    SUT_tc_dict = load_tc_from_file(SUT_equipped_test_file).all_tc
    mid = time.time()

    run_tcp(mitigated_tc_dict, SUT_tc_dict, max_instance_number=100, save_file=save_test_file)
    print(f'load time: {(mid - start)} s')
    print(f'all time: {(time.time() - start)} s')
