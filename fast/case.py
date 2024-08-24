import random

from encode import get_abstract_list, get_abstract_int
from utils import is_correct_api, get_default_args_dict, preprocess_params
from tensorflow import keras

all_layer_default_args_dict = {}  # {op1:{para1: v1, para2:v2 ...}...}

def update_all_alyer_default_args_dict(default_args_dict: dict):
    all_layer_default_args_dict.update(default_args_dict)



class TC:
    def __init__(self, line_id, test_cmd_str):
        self.id = line_id
        self.test_cmd_str = test_cmd_str
        self.is_valid = True  # if not, skip the following info
        self.layer = None
        self.all_para_info = {}
        self.abstract_tc = {}
        self.encode = None

        self.parse_test()

    def __init__(self, test_cmd_str, layer, all_para_info, abstract_tc, encode):#for torch
        self.test_cmd_str = test_cmd_str
        self.is_valid = True
        self.layer = layer
        self.all_para_info = all_para_info
        self.abstract_tc = abstract_tc
        self.encode = encode

    def __lt__(self, other):
        return self.layer < other.layer
    def parse_test(self):
        # layer_test(keras.layers.Dense,args=(),kwargs={'units':32,},input_shape=[3, 784],input_dtype='float32',)
        print(f'{"-" * 20}\n>>> Test case {self.id}: ', self.test_cmd_str.strip())
        layer_args, input_str = self.test_cmd_str.split(",input_shape=")
        input_shape, input_dtype = input_str.strip()[:-2].split(",input_dtype=")
        input_shape = eval(input_shape)
        input_dtype = eval(input_dtype)  # delete the ''
        self.all_para_info['input_shape'] = input_shape
        self.all_para_info['input_dtype'] = input_dtype

        layer_args = layer_args[len("layer_test("):].strip()
        self.layer = layer_args.split(",")[0]

        args_kwargs = layer_args[len(self.layer) + 1:]
        # args=(),kwargs={'units':32,'activation':"linear",'use_bias':True,}
        args, kwargs = args_kwargs.split(",kwargs=")

        args = args[len('args=('):-1]
        # print("debug,", f"list([{args}])")
        args_list = eval(f"list([{args}])")

        for i, elem in enumerate(args_list):
            self.all_para_info[f'para_{i}'] = elem
        kwargs_dict = eval(kwargs)
        self.all_para_info.update(kwargs_dict)

        if is_correct_api(self.layer):
            if self.layer not in all_layer_default_args_dict.keys():
                default_args_dict = get_default_args_dict(eval(self.layer))  # get the complete para info using dict
                # the default args use abstract format
                default_args_dict_abstract = self.get_abstract_test_case(default_args_dict)
                all_layer_default_args_dict[self.layer] = default_args_dict_abstract
            else:
                default_args_dict = all_layer_default_args_dict[self.layer]

            # let each test case have the same para number
            processed_args = preprocess_params(default_args_dict, self.all_para_info)
            print('>>> default_args_dict:', default_args_dict)
            if processed_args:  # return False if lack value for an un-default para.
                self.all_para_info = processed_args
                self.abstract_tc = self.get_abstract_test_case(self.all_para_info)
                print('>>> processed_args:', self.abstract_tc)
                self.encode = self.abstract_tc.values()
            else:
                self.is_valid = False

        else:
            # wrong test case, skip it.
            self.is_valid = False
            return

    def parse_torch_test(self):
        # layer_test(keras.layers.Dense,args=(),kwargs={'units':32,},input_shape=[3, 784],input_dtype='float32',)
        print(f'{"-" * 20}\n>>> Test case {self.id}: ', self.test_cmd_str.strip())
        layer_args, input_str = self.test_cmd_str.split(",.eval(), input_data=")  # todo: continue...
        input_shape, input_dtype = input_str.strip()[:-2].split(",input_dtype=")
        input_shape = eval(input_shape)
        input_dtype = eval(input_dtype)  # delete the ''
        self.all_para_info['input_shape'] = input_shape
        self.all_para_info['input_dtype'] = input_dtype

        layer_args = layer_args[len("layer_test("):].strip()
        self.layer = layer_args.split(",")[0]

        args_kwargs = layer_args[len(self.layer) + 1:]
        # args=(),kwargs={'units':32,'activation':"linear",'use_bias':True,}
        args, kwargs = args_kwargs.split(",kwargs=")

        args = args[len('args=('):-1]
        args_list = eval(f"list([{args}])")

        for i, elem in enumerate(args_list):
            self.all_para_info[f'para_{i}'] = elem
        kwargs_dict = eval(kwargs)
        self.all_para_info.update(kwargs_dict)

        if is_correct_api(self.layer):
            if self.layer not in all_layer_default_args_dict.keys():
                default_args_dict = get_default_args_dict(eval(self.layer))  # get the complete para info using dict
                # the default args use abstract format
                default_args_dict_abstract = self.get_abstract_test_case(default_args_dict)
                all_layer_default_args_dict[self.layer] = default_args_dict_abstract
            else:
                default_args_dict = all_layer_default_args_dict[self.layer]

            # let each test case have the same para number
            processed_args = preprocess_params(default_args_dict, self.all_para_info)
            print('>>> default_args_dict:', default_args_dict)
            print('>>> processed_args:', processed_args)
            if processed_args:  # return False if lack value for an un-default para.
                self.all_para_info = processed_args
            else:
                self.is_valid = False
        else:
            # wrong test case, skip it.
            self.is_valid = False
            return

    @staticmethod
    def get_abstract_test_case(_dict):
        abstract_dict = {}
        for k, v in _dict.items():
            if isinstance(v, int) or v is None:
                new_v = get_abstract_int(v)
            elif isinstance(v, list) or isinstance(v, tuple):
                new_v = get_abstract_list(v)
            elif isinstance(v, bool) or isinstance(v, str):
                new_v = v
            elif isinstance(v, float):
                new_v = 0.1 if v > 0 else -0.1
            else:
                new_v = f'{v}'
                # raise TypeError(f"The Type {type(v)} not be support yet!")
            abstract_dict[k] = new_v

        return abstract_dict


class TCDict:
    count = 0

    def __init__(self):
        self.all_tc = {}  # {op1:[ins1, ins2], }
        self.all_selected_tc = {}  # {op1: {para1: [v1, v2], para2: [v2, v3]...}..}
        self.all_selected_tc_pair = {}  # {op1: {para1_para2: [(v1, v3), (v1, v3)]...}..}

    def add(self, tc):
        if tc.layer not in self.all_selected_tc.keys():
            self.all_selected_tc[tc.layer] = {}
            self.all_selected_tc_pair[tc.layer] = {}
            vis = {}
            for k1, v1 in all_layer_default_args_dict[tc.layer].items():  # use the abstract para value
                self.all_selected_tc[tc.layer][k1] = [v1]
                for k2, v2 in tc.abstract_tc.items():
                    if k2 not in vis:
                        break
                    self.all_selected_tc_pair[tc.layer][(k1, k2)] = [(v1, v2)]
                vis[k1] = True

        if tc.layer not in self.all_tc.keys():
            self.all_tc[tc.layer] = []
        self.all_tc[tc.layer].append(tc)
        TCDict.count += 1

    def remove(self, tc):
        # delete the tc that have been selected
        self.all_tc[tc.layer].remove(tc)
        TCDict.count -= 1

    def add2selected_tc_dict(self, tc):
        # if tc.layer not in self.all_selected_tc:
        #     self.all_selected_tc[tc.layer] = {}
        vis = {}
        for k1, v1 in tc.abstract_tc.items():  # use the abstract para value
            if v1 not in self.all_selected_tc[tc.layer][k1]:
                self.all_selected_tc[tc.layer][k1].append(v1)
            for k2, v2 in tc.abstract_tc.items():
                if k2 not in vis:
                    break
                if (k1, k2) not in self.all_selected_tc_pair[tc.layer]:
                    self.all_selected_tc_pair[tc.layer][(k1, k2)] = [(v1, v2)]
                elif (v1, v2) not in self.all_selected_tc_pair[tc.layer][(k1, k2)]:
                    self.all_selected_tc_pair[tc.layer][(k1, k2)].append((v1, v2))
            vis[k1] = True

    def rank_layer(self, tvm_equipped_tc_dict=None):
        '''
        1. rank the layer according to the number of instance for each layer.
        :return:  the smaller for the equipped_div_mitigated_rate, the higher for the priority
        '''
        if tvm_equipped_tc_dict is None:
            self.all_tc = dict(sorted(self.all_tc.items(), key=lambda k_v: len(k_v[1])))
        else:
            equipped_div_mitigated_rate = {}
            for k, v in self.all_tc.items():
                if k not in tvm_equipped_tc_dict.keys():
                    equipped_div_mitigated_rate[k] = 0
                else:
                    equipped_div_mitigated_rate[k] = len(tvm_equipped_tc_dict[k]) / len(v)
            equipped_div_mitigated_rate = dict(
                sorted(equipped_div_mitigated_rate.items(), key=lambda x: x[1], reverse=True))
            self.all_tc = {key: self.all_tc[key] for key in equipped_div_mitigated_rate.keys()}
        print("len all_tc", str(len(self.all_tc)))

    def select_instance(self, layer_name):
        candidate_instance_list = self.all_tc[layer_name]
        history_selected_instance_dict = self.all_selected_tc[layer_name]
        # if len(history_selected_instance_dict) == 0:  # empty
        #     current_select_instance = random.choice(candidate_instance_list)

        # calculate the distance between each candidate instance and the group history selection.
        max_distance = 0
        max_distance_tc = candidate_instance_list[0]
        for candidate in candidate_instance_list:
            this_tc_params = candidate.abstract_tc
            # print('debug:', self.all_selected_tc[layer_name])
            this_distance = self.calc_distance(this_tc_params,
                                               self.all_selected_tc[layer_name],
                                               self.all_selected_tc_pair[layer_name])
            # print(f">>>>>>[debug] {this_distance}=== {this_tc_params}<---> {self.all_selected_tc[layer_name]}")
            if max_distance < this_distance:
                max_distance = this_distance
                max_distance_tc = candidate
        return max_distance_tc, max_distance

    @staticmethod
    def calc_distance(this_tc_dict: dict, selected_tc_dict: dict, selected_tc_pair: dict):
        # The larger the distance, the greater the difference
        num_diff_param = 0
        num_para = len(this_tc_dict)
        # print('debug:', this_tc_dict)
        for k, v in this_tc_dict.items():
            if v not in selected_tc_dict[k]:
                num_diff_param += 1
        distance = num_diff_param / num_para
        distance = max(distance, TCDict.calc_pair_distance(this_tc_dict, selected_tc_pair))
        return distance

    @staticmethod
    def calc_pair_distance(this_tc_dict: dict, selected_tc_pair: dict):
        num_diff_pair = 0
        num_pair = len(this_tc_dict) * (len(this_tc_dict) - 1) / 2
        vis = {}
        for k1, v1 in this_tc_dict.items():
            for k2, v2 in this_tc_dict.items():
                if k2 not in vis:
                    break
                if (k1, k2) not in selected_tc_pair or (v1, v2) not in selected_tc_pair[(k1, k2)]:
                    num_diff_pair += 1
            vis[k1] = True
        distance = num_diff_pair / num_pair
        return distance


if __name__ == '__main__':
    all_tc = ''''layer_test(keras.layers.Dropout,args=(0.3,),kwargs={},input_shape=[4, 2],input_dtype='float32',)
layer_test(keras.layers.Dense,args=(2,),kwargs={},input_shape=[4, 2],input_dtype='float32',)
layer_test(keras.layers.Dense,args=(10,),kwargs={},input_shape=[None, 3],input_dtype='float32',)
layer_test(keras.layers.LSTM,args=(),kwargs={},input_shape=[None, None, 6],input_dtype='float32',)'''
    tc_dict = TCDict()
    for i, tc in enumerate(all_tc.split('\n')):
        if tc.startswith('layer_test'):
            new_tc = TC(i, tc)
            tc_dict.add(new_tc)
    print(tc_dict.all_tc)
    print(tc_dict)
