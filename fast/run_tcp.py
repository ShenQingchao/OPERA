import heapq
import random
import time

from case import TC, TCDict
from load_torch import preprocess_torch_test


def run_tcp(tc_dict, tvm_equipped_tc_dict, max_instance_number=1, save_file='ranked_test_case.py'):
    tc_dict.rank_layer(tvm_equipped_tc_dict=tvm_equipped_tc_dict)
    tvm_dict = TCDict()
    all_dict = TCDict()
    rate = {}
    len_all_tc = {}
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
            for key in all_tc_dict.keys():
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
            # all_tc_dict = all_dict.all_selected_tc[layer]
            # for key in all_tc_dict.keys():
            #     r *= 1.0 / len(all_tc_dict[key])
            # all_tc_dict_pair = all_dict.all_selected_tc_pair[layer]
            # for key, val in all_tc_dict_pair.items():
            #     r *= 1.0 / len(all_tc_dict_pair[key])
            r = 0.0
        r = 1.0 - r
        print(f'{layer} {r}')
        rate[layer] = r

    for layer in all_dict.all_tc.keys():
        len_all_tc[layer] = len(all_dict.all_tc[layer])

    heap = []

    for layer_name, instance_list in tc_dict.all_tc.items():
        if len(instance_list) == 0:  # skip it if the layer group is empty
            continue
        this_selected_tc, max_distance = tc_dict.select_instance(layer_name)
        if max_distance == 0:
            continue
        heapq.heappush(heap, (-rate[layer_name] * max_distance, this_selected_tc))
    will_delete_layer_group = []
    while len(heap) != 0:

        max_distance, this_selected_tc = heapq.heappop(heap)
        # print(f'{len(heap)} {max_distance}')
        with open(save_file, 'a', encoding='utf-8') as out_f:
            out_f.write(this_selected_tc.test_cmd_str)
        tc_dict.add2selected_tc_dict(this_selected_tc)
        tc_dict.remove(this_selected_tc)
        layer_name = this_selected_tc.layer
        # print(f'DEBUG {this_selected_tc.abstract_tc} -> {tc_dict.all_selected_tc[layer_name]}')
        if max_distance == 0 or len(tc_dict.all_tc[layer_name]) == 0:
            del tc_dict.all_tc[layer_name]
            continue
        else:
            # r = 1.0
            # all_tc_dict = all_dict.all_selected_tc[layer_name]
            # selected_tc = tc_dict.all_selected_tc[layer_name]
            # for key in all_tc_dict.keys():
            #     if key in tc_dict.all_selected_tc[layer_name]:
            #         r *= 1.0 * len(selected_tc[key]) / len(all_tc_dict[key])
            #     else:
            #         r *= 1.0 / len(all_tc_dict[key])
            # all_selected_tc_pair = all_dict.all_selected_tc_pair[layer_name]
            # all_tc_dict_pair = all_dict.all_selected_tc_pair[layer_name]
            # for key, val in all_tc_dict_pair.items():
            #     if key in all_selected_tc_pair:
            #         r *= 1.0 * len(all_selected_tc_pair[key]) / len(all_tc_dict_pair[key])
            #     else:
            #         r *= 1.0 / len(all_tc_dict_pair[key])

            # r = 1.0 - r
            # rate[layer_name] = r
            # print(f'update_{layer} {r}')
            rate[layer_name] -= rate[layer_name] / len_all_tc[layer_name]
            selected_tc, distance = tc_dict.select_instance(layer_name)
            # print(f'push {selected_tc.test_cmd_str} {-rate[layer_name] * distance} {max_distance}')
            heapq.heappush(heap, (-rate[layer_name] * distance, selected_tc))

    # for selected_num in range(max_instance_number):  # selected num for each layer
    #     will_delete_layer_group = []
    #     if len(tc_dict.all_tc) == 0:
    #         break
    #     for layer_name, instance_list in tc_dict.all_tc.items():
    #         # print(f'Layer: {layer_name}')
    #         if len(instance_list) == 0:  # skip it if the layer group is empty
    #             continue
    #         this_selected_tc, max_distance = tc_dict.select_instance(layer_name)
    #         if max_distance == 0:
    #             # set the layer instance group as empty
    #             will_delete_layer_group.append(layer_name)
    #             continue
    #         tc_dict.remove(this_selected_tc)
    #         tc_dict.add2selected_tc_dict(this_selected_tc)
    #
    #         with open(save_file, 'a', encoding='utf-8') as out_f:
    #             out_f.write(this_selected_tc.test_cmd_str)
    #     # delete max_distance = 0 layer group
    #     for del_layer in will_delete_layer_group:
    #         del tc_dict.all_tc[del_layer]


def load_tc_from_file(tc_file_name):
    tc_dict = TCDict()
    with open(tc_file_name, 'r', encoding='utf-8') as intput_f:
        all_lines = intput_f.readlines()

    for i, line in enumerate(all_lines):
        if line.startswith("layer_test"):
            new_tc = TC(i, line)
            if new_tc.is_valid:  # skip the invalid test case.
                tc_dict.add(new_tc)
    return tc_dict


if __name__ == '__main__':
    # origin_test_file = "data/keras_borrow_all_test.py"
    # # origin_test_file = "/share_host/TVMFT/BorrowTests/keras/all_borrow_test.py"
    # mitigated_tc_dict = load_tc_from_file(origin_test_file)
    #
    # tvm_equipped_test_file = "data/tvm_keras_all_test.py"
    # tvm_tc_dict = load_tc_from_file(tvm_equipped_test_file).all_tc
    #
    # save_test_file = "ranked_test_case.py"
    # run_tcp(mitigated_tc_dict, tvm_tc_dict, max_instance_number=100, save_file=save_test_file)
    #

    # pytorch
    origin_test_file = "data/original_migrated_torch_tc.py.py"
    # origin_test_file = "/share_host/TVMFT/BorrowTests/keras/all_borrow_test.py"
    mitigated_tc_dict = preprocess_torch_test(origin_test_file)

    tvm_equipped_test_file = "data/tvm_torch_all_test.py"
    tvm_tc_dict = preprocess_torch_test(tvm_equipped_test_file).all_tc

    save_test_file = "torch_ranked_test_case.py"
    run_tcp(mitigated_tc_dict, tvm_tc_dict, max_instance_number=100, save_file=save_test_file)
