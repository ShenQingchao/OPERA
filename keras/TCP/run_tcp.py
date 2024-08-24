import random
import time
import os
from TCP.case import TC, TCDict
import heapq


def run_tcp(SUT, tc_dict, tvm_equipped_tc_dict, max_instance_number=1, save_file='ranked_test_case.py'):
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
    print('layers', len(tc_dict.all_tc.keys()))

    for layer_name, instance_list in tc_dict.all_tc.items():
        if len(instance_list) == 0:  # skip it if the layer group is empty
            continue
        this_selected_tc, max_distance = tc_dict.select_instance(layer_name, SUT)
        if max_distance == 0:
            continue
        heapq.heappush(heap, (-rate[layer_name] * max_distance, this_selected_tc))
    will_delete_layer_group = []
    while len(heap) != 0:

        max_distance, this_selected_tc = heapq.heappop(heap)
        print(f'{len(heap)} {max_distance}')
        with open(save_file, 'a', encoding='utf-8') as out_f:
            line_cnt = this_selected_tc.id
            out_f.write(this_selected_tc.test_cmd_str.strip()[:-2] + f",count={line_cnt},)\n")
            # out_f.write(this_selected_tc.test_cmd_str)
        tc_dict.add2selected_tc_dict(this_selected_tc)
        tc_dict.remove(this_selected_tc)
        layer_name = this_selected_tc.layer
        # print(f'DEBUG {this_selected_tc.abstract_tc} -> {tc_dict.all_selected_tc[layer_name]}')
        if max_distance == 0 or len(tc_dict.all_tc[layer_name]) == 0:
            # del tc_dict.all_tc[layer_name]
            continue
        else:
            selected_tc, distance = tc_dict.select_instance(layer_name, SUT)
            heapq.heappush(heap, (-rate[layer_name] * distance, selected_tc))

    tests = []
    for layer_name, instance_list in tc_dict.all_tc.items():
        tests.extend(instance_list)
    random.seed(1)
    random.shuffle(tests)

    print(f'len of tests {len(tests)}')
    with open(save_file, 'a', encoding='utf-8') as out_f:
        for test in tests:
            line_cnt = test.id
            out_f.write(test.test_cmd_str.strip()[:-2] + f', count={line_cnt},)\n')


def load_tc_from_file(tc_file_name):
    tc_dict = TCDict()
    with open(tc_file_name, 'r', encoding='utf-8') as intput_f:
        all_lines = intput_f.readlines()

    for i, line in enumerate(all_lines):
        if line.startswith("layer_test") or line.startswith("verify_model") or line.startswith('para_'):
            new_tc = TC(i, line)
            if new_tc.is_valid:  # skip the invalid test case.
                tc_dict.add(new_tc)
    print(len(tc_dict.all_tc.keys()))
    return tc_dict


if __name__ == '__main__':
    front = 'keras'
    SUT = "ov"  # ov, tvm, trt
    origin_test_file = f"../data/original_migrated_{front}_tc.py"
    SUT_equipped_test_file = f"../data/{SUT}_equipped_{front}_tc.py"
    save_test_file = f"../data/ranked_{front}_tc_4_{SUT}.py"
    if os.path.exists(save_test_file):
        os.remove(save_test_file)

    start = time.time()
    mitigated_tc_dict = load_tc_from_file(origin_test_file)
    SUT_tc_dict = load_tc_from_file(SUT_equipped_test_file).all_tc
    mid = time.time()

    run_tcp(SUT, mitigated_tc_dict, SUT_tc_dict, max_instance_number=100, save_file=save_test_file)
    print(f'load time: {(mid - start)} s')
    print(f'all time: {(time.time() - start)} s')
