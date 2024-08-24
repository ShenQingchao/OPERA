import random
from TCP.encode import get_abstract_list, get_abstract_int
from TCP.utils import is_correct_operator, get_onnx_op_default_args_dict, preprocess_params
import json

all_graph_default_args_dict = {}  # {op1:{para1: v1, para2:v2 ...}...}


class TC:
    def __init__(self, line_id, test_cmd_str):
        self.id = line_id + 1
        self.test_cmd_str = test_cmd_str
        self.is_valid = True  # if not, skip the following info
        self.graph = None
        self.all_para_info = {}
        self.abstract_tc = {}
        self.encode = None

        self.parse_test_from_onnx()

    def __lt__(self, other):
        return self.id < other.id
    def parse_test_from_onnx(self):
        # make_graph(op_type=Abs, kwargs={}, input_name=('x',), input_shape=([3, 4, 5],), input_dtype=('FLOAT',), output_name=('y',), output_shape=([3, 4, 5],), output_dtype=('FLOAT',))
        # print(f'{"-" * 20}\n>>> Test case {self.id}: ', self.test_cmd_str.strip())
        def get_graph_args(graph_str, kind=1):
            #graph_str = graph_str.replace(" ", "")
            graph_args_dict = {}
            graph_args, input_str, output_str = graph_str.rsplit(", input_name=", 1)[0], \
                graph_str.rsplit(", input_name=", 1)[1].rsplit(", output_name=", 1)[0], \
                graph_str.rsplit(", output_name=", 1)[1]
            # print("output_str: ", output_str)
            output_name = output_str.rsplit(", output_shape=", 1)[0]
            output_str = output_str.rsplit(", output_shape=", 1)[1].strip()[:-1]
            output_shape, output_dtype = output_str.rsplit(", output_dtype=", 1)
            # output_shape, output_dtype = output_str.strip()[:-1].split(", output_shape=")[1].split(", output_dtype=")
            input_name = input_str.rsplit(", input_shape=", 1)[0]
            input_str = input_str.rsplit(", input_shape=", 1)[1].strip()
            input_shape, input_dtype = input_str.rsplit(", input_dtype=", 1)
            input_name = eval(input_name)
            input_shape = eval(input_shape)
            input_dtype = eval(input_dtype)
            output_name = eval(output_name)
            output_shape = eval(output_shape)
            output_dtype = eval(output_dtype)
            # print(output_name, output_shape, output_dtype)
            graph_args = graph_args[len("make_graph(op_type='"):].strip()
            graph_name = graph_args.split("', ")[0]
            graph_args = graph_args[len(graph_name)+len("', kwargs="):]
            all_args_dict = eval(graph_args)
            for key, value in all_args_dict.items():
                if isinstance(value, str) and value.find("make_graph") != -1:
                    # Extract the inner dictionary from the value
                    inner_dict_str = value
                    inner_dict = get_graph_args(inner_dict_str, 2)
                    print(json.dumps(inner_dict))
                    # Update the corresponding key in all_args_dict with the extracted inner dictionary
                    all_args_dict[key] = json.dumps(inner_dict)
            if kind == 1:
                self.graph = graph_name
                self.all_para_info['input_shape'] = input_shape
                self.all_para_info['input_dtype'] = input_dtype
                self.all_para_info['output_shape'] = output_shape
                self.all_para_info['output_dtype'] = output_dtype
                self.all_para_info.update(all_args_dict)
                return
            elif kind == 2:
                graph_args_dict['op_type'] = graph_name
                graph_args_dict['input_shape'] = input_shape
                graph_args_dict['input_dtype'] = input_dtype
                graph_args_dict['output_shape'] = output_shape
                graph_args_dict['output_dtype'] = output_dtype
                graph_args_dict.update(all_args_dict)
                return graph_args_dict
        get_graph_args(self.test_cmd_str, 1)

        if is_correct_operator(self.graph):
            if self.graph not in all_graph_default_args_dict.keys():
                default_args_dict = get_onnx_op_default_args_dict(self.graph)  # get the complete para info using dict
                # the default args use abstract format
                default_args_dict_abstract = self.get_abstract_test_case(default_args_dict)
                all_graph_default_args_dict[self.graph] = default_args_dict_abstract
            else:
                default_args_dict = all_graph_default_args_dict[self.graph]


            # let each test case have the same para number
            processed_args = preprocess_params(default_args_dict, self.all_para_info)
            # print('>>> default_args_dict:', default_args_dict)
            if processed_args:  # return False if lack value for an un-default para.
                self.all_para_info = processed_args
                self.abstract_tc = self.get_abstract_test_case(self.all_para_info)
                # print('>>> processed_args:', self.abstract_tc)
                if self.graph == 'SequenceConstruct':
                    print(self.test_cmd_str)
                    print('>>> processed_args:', self.abstract_tc)
                self.encode = self.abstract_tc.values()
            # else:
            #     self.is_valid = False
            #     return

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
            elif isinstance(v, bool):
                new_v = v
            elif isinstance(v, float):
                new_v = 0.1 if v > 0 else -0.1
            elif isinstance(v, str):
                new_v = v
            elif isinstance(v, bytes):
                new_v = v.decode()
            else:
                # print("[debug]", k, v)
                raise TypeError(f"The Type {type(v)} not be support yet!")
            abstract_dict[k] = new_v

        return abstract_dict


class TCDict:
    count = 0

    def __init__(self):
        self.all_tc = {}  # {op1:[ins1, ins2], }
        self.all_selected_tc = {}  # {op1: {para1: [v1, v2], para2: [v2, v3]...}..}
        self.all_selected_tc_pair = {}  # {op1: {para1_para2: [(v1, v3), (v1, v3)]...}..}
        random.seed(0)

    def add(self, tc):
        if tc.graph not in self.all_selected_tc.keys():
            self.all_selected_tc[tc.graph] = {}
            self.all_selected_tc_pair[tc.graph] = {}
            vis = {}
            for k1, v1 in all_graph_default_args_dict[tc.graph].items():  # use the abstract para value
                self.all_selected_tc[tc.graph][k1] = [v1]
                for k2, v2 in tc.abstract_tc.items():
                    if k2 not in vis:
                        break
                    self.all_selected_tc_pair[tc.graph][(k1, k2)] = [(v1, v2)]
                vis[k1] = True

        if tc.graph not in self.all_tc.keys():
            self.all_tc[tc.graph] = []
        self.all_tc[tc.graph].append(tc)
        TCDict.count += 1

    def remove(self, tc):
        # delete the tc that have been selected
        self.all_tc[tc.graph].remove(tc)
        TCDict.count -= 1

    def add2selected_tc_dict(self, tc):
        # if tc.layer not in self.all_selected_tc:
        #     self.all_selected_tc[tc.layer] = {}
        vis = {}
        for k1, v1 in tc.abstract_tc.items():  # use the abstract para value
            if v1 not in self.all_selected_tc[tc.graph][k1]:
                self.all_selected_tc[tc.graph][k1].append(v1)
            for k2, v2 in tc.abstract_tc.items():
                if k2 not in vis:
                    break
                if (k1, k2) not in self.all_selected_tc_pair[tc.graph]:
                    self.all_selected_tc_pair[tc.graph][(k1, k2)] = [(v1, v2)]
                elif (v1, v2) not in self.all_selected_tc_pair[tc.graph][(k1, k2)]:
                    self.all_selected_tc_pair[tc.graph][(k1, k2)].append((v1, v2))
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
            if layer_name == 'SequenceConstruct':
                print(f">>>>>>[debug] {this_distance}=== {this_tc_params}<---> {self.all_selected_tc[layer_name]}")
            if max_distance < this_distance:
                max_distance = this_distance
                max_distance_tc = candidate

        return max_distance_tc, max_distance

    @staticmethod
    def calc_distance(this_tc_dict: dict, selected_tc_dict: dict, selected_tc_pair: dict):
        # The larger the distance, the greater the difference
        num_diff_param = 0.0
        num_para = len(this_tc_dict)
        # print('debug:', this_tc_dict)
        for k, v in this_tc_dict.items():
            if v not in selected_tc_dict[k]:
                if len(selected_tc_dict[k]) == 0:
                    num_diff_param += 1
                else:
                    num_diff_param += 1 / len(selected_tc_dict[k])
        if num_para == 0:
            return 0

        distance = num_diff_param / num_para
        distance = max(distance, TCDict.calc_pair_distance(this_tc_dict, selected_tc_pair))
        # distance = distance + TCDict.calc_pair_distance(this_tc_dict, selected_tc_pair)
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
                if (k1, k2) not in selected_tc_pair:
                    num_diff_pair += 1
                elif (v1, v2) not in selected_tc_pair[(k1, k2)]:
                    num_diff_pair += 1 / len(selected_tc_pair[(k1, k2)])
            vis[k1] = True
        distance = num_diff_pair / num_pair
        return distance

if __name__ == '__main__':
    all_tc = '''
make_graph(op_type=If, kwargs={'else_branch': "make_graph(op_type=Constant, kwargs={'value': '[5. 4. 3. 2. 1.]'}, input_name=(), input_shape=(), input_dtype=(), output_name=('else_out',), output_shape=([5],), output_dtype=('FLOAT',))", 'then_branch': "make_graph(op_type=Constant, kwargs={'value': '[1. 2. 3. 4. 5.]'}, input_name=(), input_shape=(), input_dtype=(), output_name=('then_out',), output_shape=([5],), output_dtype=('FLOAT',))"}, input_name=('cond',), input_shape=([],), input_dtype=('BOOL',), output_name=('res',), output_shape=([5],), output_dtype=('FLOAT',))
'''

    tc_dict = TCDict()
    for i, tc in enumerate(all_tc.split('\n')):
        if tc.startswith('make_graph'):
            new_tc = TC(i, tc)
            if new_tc.is_valid:
                tc_dict.add(new_tc)
    print(tc_dict.all_tc)
    #print(tc_dict)

