

## OPERA

<a href="https://arxiv.org/pdf/2407.16626"><img src="https://img.shields.io/badge/Paper-ICSE'25-a5fed.svg"></a>  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-a5fed.svg"></a> 

This repository provides the tool (i.e., OPERA) and all experimental data for our work: "A Tale of Two DL Cities: When Library Tests Meet Compiler", which has been accepted by ICSE'2025 . 

### Reproducibility

#### 0. File Structure

For each model format (i.e., PyTorch, Keras, and ONNX), we set up a directory in the root directory.
Each project directory includes the following items:

* **migrate**: It includes the source code for collected tests from DL libraries and converts them into DL models for testing DL compilers.
* **data**: the collected tests from DL libraries and equipped tests from DL compilers for test prioritization
* **TCP**: the test prioritization strategy of OPERA for ranking all migrated tests
* **fuzz**: script for running the collected tests to detect bugs in DL compilers


####  1. Build Environment

> Install the related DL libraries environment for collected source test cases as follows:
* PyTorch v1.7:
     `pip install torch==1.7.0`
* Keras v2.3:
    `pip install keras==2.3.1`
* ONNX 1.8:
    `pip install onnx==1.8.0`

> Install the test object as follows:
* TVM (v0.13):
     Please refer to [the official documentation](https://tvm.apache.org/docs/install/from_source.html) to install it from the source. You can execute `git checkout b48fcab` to get the same version as this work before installing it.
* OpenVINO (v2023.1.0)
     `pip install openvino==2023.1.0`
* TensorRT v8.6
     `pip install tensorrt==8.6`


#### 2. Migrate the tests
> instruments of each operator API in the source code of DL compilers for operator instance extraction.
###### Steps:
  1) execute all test suites equipped by the three DL libraries and the DocTer fuzzing under the instrumented DL libraries. Detailed instrumentation steps are shown in the `readme.md` file of each subproject.
  2) save the operator instance and wrap them into DL models automatically in the designed instrumented code,
  3) The extracted tests were saved in the path library_name/data (e.g., keras/data)


####  3. Run Test Prioritization
Execute the prioritization to rank all migrated tests.
```
python run_tcp.py
```
The result was saved in the current path (e.g., ranked_test_case_keras.py)

#### 4. Run the fuzzing
```
cd project_name/fuzz   # replace project_name with real name from [torch, keras, onnx]
# execute the fuzzing
python run_fuzz.py ../data/original_migrated_onnx_tc.py SUT_name dllibrary_name
```
> replace `SUT_name` with one of the software under tests including in [tvm, openvino, trt];
> replace `dllibrary_name` with one of the model frameworks from [torch, keras, onnx]
----

### 5. Supplement Results

You can visit the [supplement_results.md](./supplement_results.md) file to see the supplement information about our paper.

### 6. [Bug Details](./bugs.md) 

This work has detected **170** previously unknown bugs, **more than 100** of which have been confirmed/fixed by developers. The detail information can be found in the [bugs.md file](./bugs.md). 



### Citation

Please cite our paper if this work is useful for you.

```
@article{shen2024tale,
  title={A Tale of Two DL Cities: When Library Tests Meet Compiler},
  author={Shen, Qingchao and Tian, Yongqiang and Ma, Haoyang and Chen, Junjie and Huang, Lili and Fu, Ruifeng and Cheung, Shing-Chi and Wang, Zan},
  journal={arXiv preprint arXiv:2407.16626},
  year={2024}
}
```
