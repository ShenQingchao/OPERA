### Supplement Results

#### 1. RAUC-k metric

$RAUC-k$ is a metric to measure the prioritization effectiveness when all prioritized tests can not be executed completely in a limited time practically. 
Therefore, RAUC-s are proposed to measure the prioritization effectiveness when only top k tests can be executed.
Specifically, it is calculated based on the prioritization result graph with the number of tests as the x-axis and the bug number as the y-axis.
The RAUC is determined by calculating the area under the curve of the prioritization technique and contrasting it with the area under the curve of the ideal prioritization, which represents the sequential order in which the test cases would have been executed had all bugs been known beforehand.
In our study, we evaluated the performance of the TCP technique on different proportions of test cases, specifically 25\%, 50\%, 75\%, and 100\% of the total number of tests, which we referred to as RAUC-25\%, RAUC-50\%, RAUC-75\%, and RAUC-100\% respectively. A higher value of RAUC-k indicates better performance of the prioritization strategy. The bold in the table means the best value.
![image](https://github.com/AnonymousWorks/OPERA/assets/89679728/3c0e7e36-5974-4ff6-af7f-17ff2a2952f8)



#### 2. Templates of Model Generation for Three Libraries

##### PyTorch

   ![torch_template](https://github.com/AnonymousWorks/OPERA/assets/89128704/6d233eb0-0746-4711-a675-3cb3e1519b16)

#####  Keras

![keras_template](https://github.com/AnonymousWorks/OPERA/assets/89128704/10e34068-8dae-47f4-90ec-7505aa9f3744)

#####  ONNX

 ![onnx_template](https://github.com/AnonymousWorks/OPERA/assets/89128704/75b37d1d-e23d-48b5-adfa-3effe4b505d5)
