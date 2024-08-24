## A Tale of Two DL Cities: When Library Tests Meet Compiler

This repository provides the tool (i.e., OPERA) and all experimental data for an empirical study about a test migration-based idea. OPERA is a test-migration-based technique to improve the testing of the model loading stage of DL compilers. It considers different sources of test inputs in DL libraries for migration. Also, it designs a diversity-based test prioritization strategy to migrate and execute those test inputs that are more likely to detect diverse bugs in the model loading stage, to improve the testing efficiency.


### Reproducibility

#### 0. File Structure
For each model format (i.e., PyTorch, Keras, and ONNX), we set up a directory in the root directory.
Each project directory includes the following items:

* **migrate**: It includes the source code for collected test inputs from DL libraries and converts them into DL models for testing DL compilers.
* **data**: the collected test inputs from DL libraries and equipped test inputs from DL compilers for test prioritization
* **TCP**: the test prioritization strategy of OPERA for ranking all migrated test inputs
* **fuzz**: script for running the collected test inputs to detect bugs in DL compilers


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


#### 2. Migrate the test inputs
> instruments of each operator API in the source code of DL compilers for operator instance extraction.
###### Steps:
  1) execute all test suites equipped by the three DL libraries and the DocTer fuzzing under the instrumented DL libraries. Detailed instrumentation steps are shown in the `readme.md` file of each subproject.
  2) save the operator instance and wrap them into DL models automatically in the designed instrumented code,
  3) The extracted test inputs were saved in the path library_name/data (e.g., keras/data)


####  3. Run Test Prioritization
Execute the prioritization to rank all migrated test inputs.
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

### Supplement Results (RAUC-k metric)
$RAUC-k$ is a metric to measure the prioritization effectiveness when all prioritized tests can not be executed completely in a limited time practically. 
Therefore, RAUC-s are proposed to measure the prioritization effectiveness when only top k tests can be executed.
Specifically, it is calculated based on the prioritization result graph with the number of tests as the x-axis and the bug number as the y-axis.
The RAUC is determined by calculating the area under the curve of the prioritization technique and contrasting it with the area under the curve of the ideal prioritization, which represents the sequential order in which the test cases would have been executed had all bugs been known beforehand.
In our study, we evaluated the performance of the TCP technique on different proportions of test cases, specifically 25\%, 50\%, 75\%, and 100\% of the total number of tests, which we referred to as RAUC-25\%, RAUC-50\%, RAUC-75\%, and RAUC-100\% respectively. A higher value of RAUC-k indicates better performance of the prioritization strategy. The bold in the table means the best value.
![image](https://github.com/AnonymousWorks/OPERA/assets/89679728/3c0e7e36-5974-4ff6-af7f-17ff2a2952f8)




### Bug Details
> This work has detected 170 previously unknown bugs, 90 of which have been confirmed/fixed by developers.
The ID of each bug detected by OPERA will be released after the paper review due to the double-blind policy.



### Templates of Model Generation for Three Libraries

#### 1. PyTorch

   ![torch_template](https://github.com/AnonymousWorks/OPERA/assets/89128704/6d233eb0-0746-4711-a675-3cb3e1519b16)

#### 2. Keras

   ![keras_template](https://github.com/AnonymousWorks/OPERA/assets/89128704/10e34068-8dae-47f4-90ec-7505aa9f3744)

#### 3. ONNX

   ![onnx_template](https://github.com/AnonymousWorks/OPERA/assets/89128704/75b37d1d-e23d-48b5-adfa-3effe4b505d5)

