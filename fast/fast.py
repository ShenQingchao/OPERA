'''
This file is part of an ICSE'18 submission that is currently under review. 
For more information visit: https://github.com/icse18-FAST/FAST.
    
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this source.  If not, see <http://www.gnu.org/licenses/>.
'''

from collections import defaultdict
from struct import pack, unpack
import os
import random
import sys
import time
import re
import lsh


def load_torch_test(test_file):
    all_test_case_set = set()
    with open(test_file, 'r', encoding='utf-8') as test_f:
        all_lines = test_f.readlines()
    all_lines = ''.join(all_lines)
    pattern = r'# test_id: \d+\s?\n'
    all_test_cases = re.split(pattern, all_lines)
    all_test_cases = all_test_cases[1:]
    print(f'len all_test_cases {len(all_test_cases)}')
    cnt = 0
    count_case = {}
    for test_case in all_test_cases:
        cnt += 1
        test_case = test_case.strip()
        if test_case in all_test_case_set:  # deduplicate]
            print('deduplicate')
            continue
        all_test_case_set.add(test_case)
        count_case[test_case] = cnt
    print(f'len all_test_set {len(all_test_case_set)}')
    return all_test_case_set, count_case

def load_keras(test_file):
    all_test_case_set = set()
    count_case = {}
    with open(test_file, 'r', encoding='utf-8') as test_f:
        all_lines = test_f.readlines()
    cnt = 0
    for test_case in all_lines:
        cnt += 1
        test_case = test_case.strip()

        if test_case in all_test_case_set:  # deduplicate
            continue
        all_test_case_set.add(test_case)
        count_case[test_case] = cnt

    return all_test_case_set, count_case

def load_onnx(test_file):
    all_test_case_set = set()
    count_case = {}
    with open(test_file, 'r', encoding='utf-8') as test_f:
        all_lines = test_f.readlines()
    cnt = 0
    for test_case in all_lines:
        cnt += 1
        test_case = test_case.strip()

        if test_case in all_test_case_set:  # deduplicate
            continue
        all_test_case_set.add(test_case)
        count_case[test_case] = cnt

    return all_test_case_set, count_case


def loadTestSuite(front, input_file, bbox=False, k=5):
    """INPUT
    (str)input_file: path of input file

    OUTPUT
    (dict)TS: key=tc_ID, val=set(covered lines)
    """
    TS = defaultdict()
    if front == 'keras':
        tests, count_tests = load_keras(input_file)
    elif front == 'torch':
        tests, count_tests = load_torch_test(input_file)
    elif front == 'onnx':
        tests, count_tests = load_onnx(input_file)
    tests = list(tests)
    print(f'len list {len(tests)}')
    random.seed(0)
    random.shuffle(tests)
    tcID = 1
    for i in range(len(tests)):
        tc = tests[i]
        # print(tc)
        if bbox:
            TS[tcID] = tc[:-1]
        else:
            TS[tcID] = set(tc[:-1].split())
        tcID += 1
        tests[i] = tc[:-1] + f',count={count_tests[tc]},)'


    if bbox:
        TS = lsh.kShingles(TS, k)
    return TS, tests


def storeSignatures(input_file, sigfile, hashes, bbox=False, k=5):
    with open(sigfile, "w") as sigfile:
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                if bbox:
                    # shingling
                    tc_ = tc[:-1]
                    tc_shingles = set()
                    for i in range(len(tc_) - k + 1):
                        tc_shingles.add(hash(tc_[i:i + k]))

                    sig = lsh.tcMinhashing((tcID, set(tc_shingles)), hashes)
                else:
                    tc_ = tc[:-1].split()
                    sig = lsh.tcMinhashing((tcID, set(tc_)), hashes)
                for hash_ in sig:
                    sigfile.write(repr(unpack('>d', hash_)[0]))
                    sigfile.write(" ")
                sigfile.write("\n")
                tcID += 1


def loadSignatures(input_file):
    """INPUT
    (str)input_file: path of input file

    OUTPUT
    (dict)TS: key=tc_ID, val=set(covered lines), sigtime"""
    sig = {}
    start = time.time()
    with open(input_file, "r") as fin:
        tcID = 1
        for tc in fin:
            sig[tcID] = [pack('>d', float(i)) for i in tc[:-1].split()]
            tcID += 1
    return sig, time.time() - start


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# lsh + pairwise comparison with candidate set
def fast_pw(front, input_file, r, b, bbox=False, k=5, memory=False):
    """INPUT
    (str)input_file: path of input file
    (int)r: number of rows
    (int)b: number of bands
    (bool)bbox: True if BB prioritization
    (int)k: k-shingle size (for BB prioritization)
    (bool)memory: if True keep signature in memory and do not store them to file

    OUTPUT
    (list)P: prioritized test suite
    """
    n = r * b  # number of hash functions

    hashes = [lsh.hashFamily(i) for i in range(n)]
    if memory:
        test_suite, tests = loadTestSuite(front, input_file, bbox=bbox, k=k)
        # generate minhashes signatures
        mh_t = time.time()
        tcs_minhashes = {tc[0]: lsh.tcMinhashing(tc, hashes)
                         for tc in test_suite.items()}

        mh_time = time.time() - mh_t
        ptime_start = time.time()

    else:
        # loading input file and generating minhashes signatures
        sigfile = input_file.replace(".py", ".sig")
        sigtimefile = "{}_sigtime.txt".format(input_file.split(".")[0])
        if not os.path.exists(sigfile):
            mh_t = time.time()
            storeSignatures(input_file, sigfile, hashes, bbox, k)
            mh_time = time.time() - mh_t
            with open(sigtimefile, "w") as fout:
                fout.write(repr(mh_time))
        else:
            with open(sigtimefile, "r") as fin:
                mh_time = eval(fin.read().replace("\n", ""))

        ptime_start = time.time()
        tcs_minhashes, load_time = loadSignatures(sigfile)

    tcs = set(tcs_minhashes.keys())

    BASE = 0.5
    SIZE = int(len(tcs)*BASE) + 1

    bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)

    prioritized_tcs = [0]

    # First TC
    selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
    first_tc = random.choice(list(tcs_minhashes.keys()))
    for i in range(n):
        if tcs_minhashes[first_tc][i] < selected_tcs_minhash[i]:
            selected_tcs_minhash[i] = tcs_minhashes[first_tc][i]
    prioritized_tcs.append(first_tc)
    tcs -= set([first_tc])
    del tcs_minhashes[first_tc]

    iteration, total = 0, float(len(tcs_minhashes))
    while len(tcs_minhashes) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))
            sys.stdout.flush()

        if len(tcs_minhashes) < SIZE:
            bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)
            SIZE = int(SIZE*BASE) + 1

        sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash),
                                     b, r, n)
        filtered_sim_cand = sim_cand.difference(prioritized_tcs)
        candidates = tcs - filtered_sim_cand

        if len(candidates) == 0:
            selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
            sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash),
                                         b, r, n)
            filtered_sim_cand = sim_cand.difference(prioritized_tcs)
            candidates = tcs - filtered_sim_cand
            if len(candidates) == 0:
                candidates = tcs_minhashes.keys()

        selected_tc, max_dist = random.choice(tuple(candidates)), -1
        for candidate in tcs_minhashes:
            if candidate in candidates:
                dist = lsh.jDistanceEstimate(
                    selected_tcs_minhash, tcs_minhashes[candidate])
                if dist > max_dist:
                    selected_tc, max_dist = candidate, dist

        for i in range(n):
            if tcs_minhashes[selected_tc][i] < selected_tcs_minhash[i]:
                selected_tcs_minhash[i] = tcs_minhashes[selected_tc][i]

        prioritized_tcs.append(selected_tc)
        tcs -= set([selected_tc])
        del tcs_minhashes[selected_tc]

    ptime = time.time() - ptime_start
    print(f'mh time {mh_time}')
    print(f'ptime {ptime}')
    with open(f'fast_{front}.py', 'w') as fast:

        for test in prioritized_tcs[1:]:
            print(tests[test-1])
            fast.write(tests[test-1] + '\n')
    return mh_time, ptime, prioritized_tcs[1:]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    start = time.time()
    # fast_pw('onnx', 'data/original_migrated_onnx_tc.py', 1, 10, memory=True)
    # fast_pw('keras', 'data/original_migrated_keras_tc.py', 1, 10, memory=True)
    fast_pw('torch', 'data/original_migrated_torch_tc.py', 1, 10, memory=True)
    print(f'all time: {time.time() - start}')
