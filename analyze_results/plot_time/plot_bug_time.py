from matplotlib import pyplot as plot
import math
import matplotlib

COLORs = {"Random": "#F08080",
      "OPERA": "#1e90ff"}

LS = {"Random": ":",
      "OPERA": "-"}


def get_accumulate_bug_num(bug_line_list, max_time):
    cumulative_bug_list = []
    cumulative_bug_num = 0
    for i in range(int(max_time+1)):
        if i in bug_line_list:
            cumulative_bug_num += bug_line_list.count(i)
        cumulative_bug_list.append(cumulative_bug_num)
    return cumulative_bug_list


def plot_all(all_method_res_dict, sut, project, max_time):
    ax.set_ylabel("# Bugs", ) # fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    for method, cumulative_bug_list in all_method_res_dict.items():
        bugs_id_list = list(range(len(cumulative_bug_list)))
        if project == "onnx":
            bugs_id_list_mins = list(range(math.ceil(len(cumulative_bug_list) / 60)))
            cumulative_bug_list_mins = [cumulative_bug_list[j * 60] for j in range(0, len(bugs_id_list_mins) - 1)]
            cumulative_bug_list_mins.append(cumulative_bug_list[-1])
            ax.plot(bugs_id_list_mins, cumulative_bug_list_mins,  label=method, linewidth=3, ls=LS[method], color=COLORs[method])
        else:
            bugs_id_list_hour = list(range(math.ceil(len(cumulative_bug_list) / 3600)))
            cumulative_bug_list_hour = [cumulative_bug_list[j * 3600] for j in range(0, len(bugs_id_list_hour) - 1)]
            cumulative_bug_list_hour.append(cumulative_bug_list[-1])
            ax.plot(bugs_id_list_hour, cumulative_bug_list_hour,  label=method, linewidth=3, ls=LS[method], color=COLORs[method])
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='plain', axis='x')
    if sut == 'tvm':
        if project == "torch":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 61])
            ax.set_ylim([-1, 33])
        elif project == "keras":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 70])
            ax.set_ylim([-1, 38])
        elif project == "onnx":
            ax.set_xlabel("Time (min)", )  # fontweight='bold')
            ax.set_xlim([-1, 35])
            ax.set_ylim([-1, 16])
    elif sut == 'ov':
        if project == "torch":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 61])
            ax.set_ylim([-1, 28])
        elif project == "keras":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 71])
            ax.set_ylim([-1, 12])
        elif project == "onnx":
            ax.set_xlabel("Time (min)", )  # fontweight='bold')
            ax.set_xlim([-1, 45])
            ax.set_ylim([-1, 14])
    elif sut == 'trt':
        if project == "torch":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 151])
            ax.set_ylim([-1, 36])
        elif project == "onnx":
            ax.set_xlabel("Time (min)", )  # fontweight='bold')
            ax.set_xlim([-1, 100])
            ax.set_ylim([-1, 14])
    if sut == 'trt':
        sut = 'TensorRT'
    elif sut == 'tvm':
        sut = 'TVM'
    elif sut == 'ov':
        sut = 'OpenVINO'
    if project == 'torch':
        project = 'PyTorch'
    elif project == 'keras':
        project = 'Keras'
    elif project == 'onnx':
        project = 'ONNX'
    title = f'({chr(ord("a")+cnt)}) {sut}-{project}'
    ax.set_title(title, fontsize=20)


def load_time_record(time_file, total_test_num):
    test_time_list = []
    with open(time_file, 'r') as time_f:
        all_lines = time_f.readlines()
    all_lines_dict = {}
    for line in all_lines:
        temp = line.strip().split('\t')
        test_id = temp[0]
        all_lines_dict[int(test_id)] = float(temp[1])
    all_lines_dict_order = dict(sorted(all_lines_dict.items(), key=lambda x: x[0]))

    test_id = 0
    accumulate_time = 0
    for test_id, time_consuming in all_lines_dict_order.items():
        time_consuming = time_consuming
        this_time = time_consuming + accumulate_time
        test_time_list.append(this_time)

        accumulate_time = this_time
        # print(this_time)

    end_test_id = int(test_id)+1
    res_test_num = total_test_num - end_test_id
    # print(end_test_id, total_test_num)
    if end_test_id < total_test_num:
        for rest_test_id in range(res_test_num):
            test_time_list.append(accumulate_time + rest_test_id*2)
            # print(before_time + rest_test_id*2)
    # print(test_time_list)
    test_time_list = [int(i) for i in test_time_list]
    return test_time_list


bug_position_tvm_dict = {
'torch_our':[25, 40, 52, 72, 83, 89, 92, 94, 122, 136, 185, 211, 227, 270, 323, 437, 463, 542, 596, 619, 623, 644, 984, 1050, 1681, 3094, 3477, 4496, 6377, 18977],
'torch_random':[34, 56, 91, 139, 276, 321, 413, 478, 524, 636, 668, 888, 1226, 1385, 1507, 1658, 2227, 3774, 4360, 5335, 6254, 7135, 8150, 10498, 12033, 17581, 22134, 28705, 47553, 67485],

'keras_our':[16, 71, 78, 79, 114, 128, 130, 145, 147, 241, 261, 299, 300, 336, 414, 444, 458, 465, 510, 577, 606, 640, 678, 716, 825, 846, 872, 946, 984, 1159, 1354, 1486, 13134, 21138],
'keras_random':[4, 11, 23, 32, 55, 98, 162, 215, 252, 304, 368, 418, 521, 649, 927, 1077, 1362, 1674, 1979, 2413, 2956, 3487, 3848, 4649, 5770, 6643, 8703, 9881, 12997, 15781, 19810, 24923, 32101, 40446],

'onnx_our':[20, 39, 87, 125, 174, 208, 288, 291, 309, 327, 427, 493, 531, 566],
'onnx_random':[31, 86, 142, 174, 227, 279, 369, 418, 500, 590, 671, 751, 819, 947],

}

bug_position_ov_dict = {
'torch_our':[9, 19, 20, 59, 106, 230, 246, 394, 419, 496, 610, 785, 786, 1049, 1293, 1543, 1747, 1873, 2633, 2826, 3358, 4868, 5777, 9526],
'torch_random':[43, 73, 127, 313, 550, 718, 1270, 1784, 2177, 2912, 4129, 5573, 6329, 7040, 7934, 9451, 10871, 14827, 16569, 19893, 24270, 38896, 50319, 60967],

'keras_our':[1, 38, 44, 62, 79, 125, 534, 1793, 2993, 11113, 20731],
'keras_random':[88, 144, 274, 1143, 3096, 4835, 8219, 10673, 14125, 22169, 38943],


'onnx_our':[11, 28, 35, 78, 81, 87, 99, 154, 157, 284, 355, 521, 537],
'onnx_random':[26, 44, 73, 88, 135, 188, 221, 288, 345, 426, 514, 632, 761],
}

bug_position_trt_dict = {
'torch_our':[8, 9, 15, 31, 41, 90, 113, 121, 144, 153, 161, 166, 240, 246, 253, 257, 263, 301, 330, 350, 374, 384, 458, 470, 725, 1525, 3316, 4180, 5759, 12874, 17467, 17914],
'torch_random':[8, 88, 177, 272, 382, 529, 587, 831, 1032, 1252, 1657, 1984, 2338, 2849, 3365, 4168, 4680, 5754, 6876, 8520, 9504, 11545, 14487, 16680, 18981, 23071, 28878, 34842, 41187, 46815, 56833, 64231],

'onnx_our':[72, 85, 93, 113, 148, 179, 234, 240, 334, 371, 402, 560],
'onnx_random':[67, 80, 125, 161, 217, 275, 310, 411, 508, 596, 703, 867],

}
if __name__ == '__main__':
    config = {
        "font.family": "sans-serif",  # 使用衬线体
        "font.sans-serif": ["Helvetica"],  # 全局默认使用衬线宋体,
        "font.size": 22,
        "axes.unicode_minus": False,
        # "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }
    matplotlib.rcParams['xtick.labelsize'] = 15
    plot.rcParams.update(config)
    fig, axs = plot.subplots(nrows=2, ncols=4, figsize=(22, 6))

    SUT_list = ['tvm', 'trt', 'ov']  # tvm, ov, trt
    cnt = -1
    for i, sut in enumerate(SUT_list):
        bug_position_dict = eval(f"bug_position_{sut}_dict")
        projects = ['torch', 'keras', 'onnx']
        all_res_dict = {}
        for j, project in enumerate(projects):
            if project == 'keras':
                test_num = 62976
                tcp_time = 18
                if sut == 'trt':
                    continue
            elif project == 'torch':
                test_num = 97134
                tcp_time = 86
            elif project == 'onnx':
                test_num = 1013
                tcp_time = 10
            cnt += 1
            bug_time_relation = load_time_record(f'time_record_{sut}_{project}.txt', test_num)
            bug_time_relation = [i+tcp_time for i in bug_time_relation]

            # bug_time_relation = [i/3600 for i in bug_time_relation]
            our_each_bug_time = [bug_time_relation[i] for i in bug_position_dict[f'{project}_our']]
            all_bug_found_total_time = our_each_bug_time[-1] / 3600
            print(f"Total time of OPERA for {project} is {all_bug_found_total_time} hours")

            random_each_bug_time = [bug_time_relation[int(i)] for i in bug_position_dict[f'{project}_random']]
            all_bug_found_total_time = random_each_bug_time[-1] / 3600
            print(f"Total time of Random for {project} is {all_bug_found_total_time} hours")
            max_time = bug_time_relation[-1]
            print(f"Total time for all {project} tests {max_time/3600} hours\n")
            # print(our_each_bug_time)
            # print(random_each_bug_time)
            all_res_dict['OPERA'] = get_accumulate_bug_num(our_each_bug_time, max_time)
            all_res_dict['Random'] = get_accumulate_bug_num(random_each_bug_time, max_time)
            # all_res_dict['OPERA$_{random}$'] = get_accumulate_bug_num(random_each_bug_time, max_time)
            pos_x = cnt // 4
            pos_y = cnt % 4
            ax = axs[pos_x][pos_y]
            plot_all(all_res_dict, sut, project, max_time=max_time)

    plot.tight_layout()
    plot.legend(loc='lower right', fontsize=20)
    plot.savefig(f"trends_all.pdf")
    plot.show()
