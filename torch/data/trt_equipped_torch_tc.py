# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 111, 111], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 111, 111], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 111, 111], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=True,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True,).eval(), input_data=[torch.randn([1, 64, 111, 111], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 55, 55], dtype=torch.float16)
para_1 = torch.randn([16, 64, 1, 1], dtype=torch.float16)
para_2 = torch.randn([16], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 64, 55, 55], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 16, 55, 55], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 16, 55, 55], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 16, 55, 55], dtype=torch.float16)
para_1 = torch.randn([64, 16, 1, 1], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 16, 55, 55], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 55, 55], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 55, 55], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 16, 55, 55], dtype=torch.float16)
para_1 = torch.randn([64, 16, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([96, 3, 7, 7], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 15 
para_0 = torch.randn([1, 128, 55, 55], dtype=torch.float16)
para_1 = torch.randn([16, 128, 1, 1], dtype=torch.float16)
para_2 = torch.randn([16], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 16 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 128, 55, 55], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 96, 109, 109], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 96, 109, 109], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 96, 109, 109], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=True,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True,).eval(), input_data=[torch.randn([1, 96, 109, 109], dtype=torch.float16)])
# test_id: 17 
para_0 = torch.randn([1, 128, 55, 55], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=True,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 18 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True,).eval(), input_data=[torch.randn([1, 128, 55, 55], dtype=torch.float16)])
# test_id: 19 
para_0 = torch.randn([1, 128, 27, 27], dtype=torch.float16)
para_1 = torch.randn([32, 128, 1, 1], dtype=torch.float16)
para_2 = torch.randn([32], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 20 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 128, 27, 27], dtype=torch.float16)])
# test_id: 21 
para_0 = torch.randn([1, 32, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 22 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 32, 27, 27], dtype=torch.float16)])
# test_id: 23 
para_0 = torch.randn([1, 32, 27, 27], dtype=torch.float16)
para_1 = torch.randn([128, 32, 1, 1], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 32, 27, 27], dtype=torch.float16)])
# test_id: 25 
para_0 = torch.randn([1, 128, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 27, 27], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 32, 27, 27], dtype=torch.float16)
para_1 = torch.randn([128, 32, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 28 
para_0 = torch.randn([1, 256, 27, 27], dtype=torch.float16)
para_1 = torch.randn([32, 256, 1, 1], dtype=torch.float16)
para_2 = torch.randn([32], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 256, 27, 27], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 256, 27, 27], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=True,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True,).eval(), input_data=[torch.randn([1, 256, 27, 27], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 256, 13, 13], dtype=torch.float16)
para_1 = torch.randn([48, 256, 1, 1], dtype=torch.float16)
para_2 = torch.randn([48], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 256, 13, 13], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 48, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 48, 13, 13], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 48, 13, 13], dtype=torch.float16)
para_1 = torch.randn([192, 48, 1, 1], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 48, 13, 13], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 192, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 13, 13], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 48, 13, 13], dtype=torch.float16)
para_1 = torch.randn([192, 48, 3, 3], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
para_0 = torch.randn([1, 384, 13, 13], dtype=torch.float16)
para_1 = torch.randn([48, 384, 1, 1], dtype=torch.float16)
para_2 = torch.randn([48], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 384, 13, 13], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 384, 13, 13], dtype=torch.float16)
para_1 = torch.randn([64, 384, 1, 1], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
para_0 = torch.randn([1, 64, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 13, 13], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 64, 13, 13], dtype=torch.float16)
para_1 = torch.randn([256, 64, 1, 1], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 64, 13, 13], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 256, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 13, 13], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 96, 54, 54], dtype=torch.float16)
para_1 = torch.randn([16, 96, 1, 1], dtype=torch.float16)
para_2 = torch.randn([16], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 96, 54, 54], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 16, 54, 54], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 16, 54, 54], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 64, 13, 13], dtype=torch.float16)
para_1 = torch.randn([256, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 51 
para_0 = torch.randn([1, 512, 13, 13], dtype=torch.float16)
para_1 = torch.randn([64, 512, 1, 1], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 52 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 512, 13, 13], dtype=torch.float16)])
# test_id: 53 
para_0 = torch.randn([1, 512, 13, 13], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 54 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 512, 13, 13], dtype=torch.float16)])
# test_id: 55 
para_0 = torch.randn([1, 512, 13, 13], dtype=torch.float16)
para_1 = torch.randn([1000, 512, 1, 1], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 56 
para_0 = torch.randn([1, 1000, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1000, 13, 13], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 1000, 13, 13], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 59 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 1000, 13, 13], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 16, 54, 54], dtype=torch.float16)
para_1 = torch.randn([64, 16, 1, 1], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 16, 54, 54], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 54, 54], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 54, 54], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 16, 54, 54], dtype=torch.float16)
para_1 = torch.randn([64, 16, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 15 
para_0 = torch.randn([1, 128, 54, 54], dtype=torch.float16)
para_1 = torch.randn([16, 128, 1, 1], dtype=torch.float16)
para_2 = torch.randn([16], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 16 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 128, 54, 54], dtype=torch.float16)])
# test_id: 17 
para_0 = torch.randn([1, 128, 54, 54], dtype=torch.float16)
para_1 = torch.randn([32, 128, 1, 1], dtype=torch.float16)
para_2 = torch.randn([32], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 18 
para_0 = torch.randn([1, 32, 54, 54], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 32, 54, 54], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 32, 54, 54], dtype=torch.float16)
para_1 = torch.randn([128, 32, 1, 1], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 32, 54, 54], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 128, 54, 54], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 54, 54], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 32, 54, 54], dtype=torch.float16)
para_1 = torch.randn([128, 32, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
para_0 = torch.randn([1, 256, 54, 54], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=True,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True,).eval(), input_data=[torch.randn([1, 256, 54, 54], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 256, 27, 27], dtype=torch.float16)
para_1 = torch.randn([32, 256, 1, 1], dtype=torch.float16)
para_2 = torch.randn([32], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 256, 27, 27], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 32, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 32, 27, 27], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 32, 27, 27], dtype=torch.float16)
para_1 = torch.randn([128, 32, 1, 1], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 32, 27, 27], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 128, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 27, 27], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 32, 27, 27], dtype=torch.float16)
para_1 = torch.randn([128, 32, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 36 
para_0 = torch.randn([1, 256, 27, 27], dtype=torch.float16)
para_1 = torch.randn([48, 256, 1, 1], dtype=torch.float16)
para_2 = torch.randn([48], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 48, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 38 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 48, 27, 27], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 39 
para_0 = torch.randn([1, 48, 27, 27], dtype=torch.float16)
para_1 = torch.randn([192, 48, 1, 1], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 40 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 48, 27, 27], dtype=torch.float16)])
# test_id: 41 
para_0 = torch.randn([1, 192, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 27, 27], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 48, 27, 27], dtype=torch.float16)
para_1 = torch.randn([192, 48, 3, 3], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
para_0 = torch.randn([1, 384, 27, 27], dtype=torch.float16)
para_1 = torch.randn([48, 384, 1, 1], dtype=torch.float16)
para_2 = torch.randn([48], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 384, 27, 27], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 384, 27, 27], dtype=torch.float16)
para_1 = torch.randn([64, 384, 1, 1], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 47 
para_0 = torch.randn([1, 64, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 48 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 27, 27], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 15 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 16 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 17 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 18 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 49 
para_0 = torch.randn([1, 64, 27, 27], dtype=torch.float16)
para_1 = torch.randn([256, 64, 1, 1], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 50 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 64, 27, 27], dtype=torch.float16)])
# test_id: 51 
para_0 = torch.randn([1, 256, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 52 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 27, 27], dtype=torch.float16)])
# test_id: 19 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 20 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 53 
para_0 = torch.randn([1, 64, 27, 27], dtype=torch.float16)
para_1 = torch.randn([256, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 54 
para_0 = torch.randn([1, 512, 27, 27], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=True,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True,).eval(), input_data=[torch.randn([1, 512, 27, 27], dtype=torch.float16)])
# test_id: 21 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 22 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 23 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 25 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 512, 13, 13], dtype=torch.float16)
para_1 = torch.randn([64, 512, 1, 1], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 512, 13, 13], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 64, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 59 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 13, 13], dtype=torch.float16)])
# test_id: 60 
para_0 = torch.randn([1, 64, 13, 13], dtype=torch.float16)
para_1 = torch.randn([256, 64, 1, 1], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 61 
verify_model(torch.nn.Conv2d(512,1000,kernel_size=1,).eval(), input_data=[torch.randn([1, 64, 13, 13], dtype=torch.float16)])
# test_id: 62 
para_0 = torch.randn([1, 256, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 63 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 13, 13], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 64 
para_0 = torch.randn([1, 64, 13, 13], dtype=torch.float16)
para_1 = torch.randn([256, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 65 
para_0 = torch.randn([1, 512, 13, 13], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 66 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 512, 13, 13], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 31 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 67 
para_0 = torch.randn([1, 512, 13, 13], dtype=torch.float16)
para_1 = torch.randn([1000, 512, 1, 1], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 68 
para_0 = torch.randn([1, 1000, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 69 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1000, 13, 13], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 70 
para_0 = torch.randn([1, 1000, 13, 13], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 71 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 1000, 13, 13], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512], dtype=torch.float16)
para_1 = torch.randn([1000, 512], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Linear(512,1000,).eval(), input_data=[torch.randn([1, 512], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 15 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 16 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 17 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 18 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 19 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 20 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 21 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 22 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 23 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 25 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 30 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 31 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 18 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 96, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 39 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
para_1 = torch.randn([224], dtype=torch.float16)
para_2 = torch.randn([224], dtype=torch.float16)
para_3 = torch.randn([224], dtype=torch.float16)
para_4 = torch.randn([224], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 40 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 41 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 224, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 45 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 46 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 47 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 48 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 49 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 50 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 51 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 52 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 53 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 54 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 55 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 56 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 57 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 58 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 59 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 60 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 61 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 62 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 63 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 65 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 66 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 67 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 68 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 69 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 70 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 71 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 72 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
para_1 = torch.randn([224], dtype=torch.float16)
para_2 = torch.randn([224], dtype=torch.float16)
para_3 = torch.randn([224], dtype=torch.float16)
para_4 = torch.randn([224], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 73 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 74 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 75 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512], dtype=torch.float16)
para_1 = torch.randn([1000, 512], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Linear(512,1000,).eval(), input_data=[torch.randn([1, 512], dtype=torch.float16)])
# test_id: 76 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 224, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 77 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 78 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 79 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 80 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 85 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 86 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 87 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 88 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 89 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 90 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 91 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 92 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 93 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 94 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 95 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 96 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
para_1 = torch.randn([352], dtype=torch.float16)
para_2 = torch.randn([352], dtype=torch.float16)
para_3 = torch.randn([352], dtype=torch.float16)
para_4 = torch.randn([352], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 97 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 98 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 99 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 100 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 352, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 101 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 102 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 103 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 104 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 105 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 106 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 107 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 108 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
para_1 = torch.randn([416], dtype=torch.float16)
para_2 = torch.randn([416], dtype=torch.float16)
para_3 = torch.randn([416], dtype=torch.float16)
para_4 = torch.randn([416], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 109 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 110 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 111 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 112 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 416, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 113 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 114 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
para_1 = torch.randn([448], dtype=torch.float16)
para_2 = torch.randn([448], dtype=torch.float16)
para_3 = torch.randn([448], dtype=torch.float16)
para_4 = torch.randn([448], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 115 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 116 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 117 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 118 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 448, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 119 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 120 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 121 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 122 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 123 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 124 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 125 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 126 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 127 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 128 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 129 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 130 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 131 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 11, 11], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (4, 4)
para_4 = (2, 2)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(256,256,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 132 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 133 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 134 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 135 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 136 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 137 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 138 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 139 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 140 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 141 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 142 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 143 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 144 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 145 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 146 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 147 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 148 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 149 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 150 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 151 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 152 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 153 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 154 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 2 
para_0 = torch.randn([1, 64, 55, 55], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 155 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 3 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 55, 55], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 55, 55], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,).eval(), input_data=[torch.randn([1, 64, 55, 55], dtype=torch.float16)])
# test_id: 156 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 157 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 158 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
para_1 = torch.randn([352], dtype=torch.float16)
para_2 = torch.randn([352], dtype=torch.float16)
para_3 = torch.randn([352], dtype=torch.float16)
para_4 = torch.randn([352], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 159 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 160 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 161 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 162 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 352, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 163 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 164 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 165 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 166 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 167 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 168 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 169 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 170 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
para_1 = torch.randn([416], dtype=torch.float16)
para_2 = torch.randn([416], dtype=torch.float16)
para_3 = torch.randn([416], dtype=torch.float16)
para_4 = torch.randn([416], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 171 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 172 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 173 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 174 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 416, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 175 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 176 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
para_1 = torch.randn([448], dtype=torch.float16)
para_2 = torch.randn([448], dtype=torch.float16)
para_3 = torch.randn([448], dtype=torch.float16)
para_4 = torch.randn([448], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 177 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 178 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 179 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 180 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 448, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 181 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 182 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 183 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 184 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 185 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 27, 27], dtype=torch.float16)
para_1 = torch.randn([192, 64, 5, 5], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(256,256,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 27, 27], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 192, 27, 27], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 27, 27], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 192, 27, 27], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,).eval(), input_data=[torch.randn([1, 192, 27, 27], dtype=torch.float16)])
# test_id: 186 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 187 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 188 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 189 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 190 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 191 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 192 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 193 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 194 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
para_1 = torch.randn([544], dtype=torch.float16)
para_2 = torch.randn([544], dtype=torch.float16)
para_3 = torch.randn([544], dtype=torch.float16)
para_4 = torch.randn([544], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 195 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 196 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 197 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 192, 13, 13], dtype=torch.float16)
para_1 = torch.randn([384, 192, 3, 3], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(256,256,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 192, 13, 13], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 384, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 13, 13], dtype=torch.float16)])
# test_id: 198 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 544, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 199 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 200 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 201 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 202 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 203 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 204 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 205 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 384, 13, 13], dtype=torch.float16)
para_1 = torch.randn([256, 384, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(256,256,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 384, 13, 13], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 13, 13], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 13, 13], dtype=torch.float16)])
# test_id: 206 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
para_1 = torch.randn([608], dtype=torch.float16)
para_2 = torch.randn([608], dtype=torch.float16)
para_3 = torch.randn([608], dtype=torch.float16)
para_4 = torch.randn([608], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 207 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 208 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 209 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 210 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 608, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 211 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 212 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = torch.randn([640], dtype=torch.float16)
para_2 = torch.randn([640], dtype=torch.float16)
para_3 = torch.randn([640], dtype=torch.float16)
para_4 = torch.randn([640], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 213 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 214 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 215 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 13, 13], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(256,256,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 13, 13], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 13, 13], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,).eval(), input_data=[torch.randn([1, 256, 13, 13], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 6, 6], dtype=torch.float16)
para_1 = (6, 6)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.AdaptiveAvgPool2d((6, 6),).eval(), input_data=[torch.randn([1, 256, 6, 6], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 9216], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 9216], dtype=torch.float16)])
# test_id: 216 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 640, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 217 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 218 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 219 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 220 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 221 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 222 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 223 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 224 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
para_1 = torch.randn([704], dtype=torch.float16)
para_2 = torch.randn([704], dtype=torch.float16)
para_3 = torch.randn([704], dtype=torch.float16)
para_4 = torch.randn([704], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 225 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 226 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 227 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 228 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 704, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 229 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 230 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
para_1 = torch.randn([736], dtype=torch.float16)
para_2 = torch.randn([736], dtype=torch.float16)
para_3 = torch.randn([736], dtype=torch.float16)
para_4 = torch.randn([736], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 231 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 232 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 233 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 234 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 736, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 235 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 236 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 237 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 238 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 239 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 240 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 241 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 242 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
para_1 = torch.randn([800], dtype=torch.float16)
para_2 = torch.randn([800], dtype=torch.float16)
para_3 = torch.randn([800], dtype=torch.float16)
para_4 = torch.randn([800], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 243 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 244 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 245 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 246 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 800, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 247 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 248 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
para_1 = torch.randn([832], dtype=torch.float16)
para_2 = torch.randn([832], dtype=torch.float16)
para_3 = torch.randn([832], dtype=torch.float16)
para_4 = torch.randn([832], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 15 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 249 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 250 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 251 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 252 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 832, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 253 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 254 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([864], dtype=torch.float16)
para_2 = torch.randn([864], dtype=torch.float16)
para_3 = torch.randn([864], dtype=torch.float16)
para_4 = torch.randn([864], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 255 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 256 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 257 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 258 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 864, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 259 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 260 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = torch.randn([896], dtype=torch.float16)
para_2 = torch.randn([896], dtype=torch.float16)
para_3 = torch.randn([896], dtype=torch.float16)
para_4 = torch.randn([896], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 261 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 262 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 263 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 264 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 896, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 265 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 266 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
para_1 = torch.randn([928], dtype=torch.float16)
para_2 = torch.randn([928], dtype=torch.float16)
para_3 = torch.randn([928], dtype=torch.float16)
para_4 = torch.randn([928], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 267 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 268 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 269 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 270 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 928, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 271 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 272 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 273 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 274 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 275 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 276 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 277 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 278 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
para_1 = torch.randn([992], dtype=torch.float16)
para_2 = torch.randn([992], dtype=torch.float16)
para_3 = torch.randn([992], dtype=torch.float16)
para_4 = torch.randn([992], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 279 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 280 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 281 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 282 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 992, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 283 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 284 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 285 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 286 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 287 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 288 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 289 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 9216], dtype=torch.float16)
para_1 = torch.randn([4096, 9216], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 290 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 9216], dtype=torch.float16)])
# test_id: 291 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 292 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 32 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 293 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 294 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 295 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 23 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 25 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 296 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 297 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 28 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 298 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 29 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 299 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 30 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 300 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 301 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 302 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 303 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 304 
para_0 = torch.randn([1, 544, 7, 7], dtype=torch.float16)
para_1 = torch.randn([544], dtype=torch.float16)
para_2 = torch.randn([544], dtype=torch.float16)
para_3 = torch.randn([544], dtype=torch.float16)
para_4 = torch.randn([544], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 305 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 544, 7, 7], dtype=torch.float16)])
# test_id: 306 
para_0 = torch.randn([1, 544, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 307 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 544, 7, 7], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 308 
para_0 = torch.randn([1, 544, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 544, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 309 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 544, 7, 7], dtype=torch.float16)])
# test_id: 310 
para_0 = torch.randn([1, 576, 7, 7], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 311 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 312 
para_0 = torch.randn([1, 576, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 313 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([512, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 314 
para_0 = torch.randn([1, 576, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 315 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 316 
para_0 = torch.randn([1, 608, 7, 7], dtype=torch.float16)
para_1 = torch.randn([608], dtype=torch.float16)
para_2 = torch.randn([608], dtype=torch.float16)
para_3 = torch.randn([608], dtype=torch.float16)
para_4 = torch.randn([608], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 317 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 608, 7, 7], dtype=torch.float16)])
# test_id: 318 
para_0 = torch.randn([1, 608, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 319 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 608, 7, 7], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 320 
para_0 = torch.randn([1, 608, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 608, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 321 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 608, 7, 7], dtype=torch.float16)])
# test_id: 322 
para_0 = torch.randn([1, 640, 7, 7], dtype=torch.float16)
para_1 = torch.randn([640], dtype=torch.float16)
para_2 = torch.randn([640], dtype=torch.float16)
para_3 = torch.randn([640], dtype=torch.float16)
para_4 = torch.randn([640], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 323 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 324 
para_0 = torch.randn([1, 640, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 325 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 326 
para_0 = torch.randn([1, 640, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 640, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 327 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 328 
para_0 = torch.randn([1, 672, 7, 7], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 329 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 330 
para_0 = torch.randn([1, 672, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 331 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 332 
para_0 = torch.randn([1, 672, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 333 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 334 
para_0 = torch.randn([1, 704, 7, 7], dtype=torch.float16)
para_1 = torch.randn([704], dtype=torch.float16)
para_2 = torch.randn([704], dtype=torch.float16)
para_3 = torch.randn([704], dtype=torch.float16)
para_4 = torch.randn([704], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 335 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 336 
para_0 = torch.randn([1, 704, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 337 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 338 
para_0 = torch.randn([1, 704, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 704, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 339 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 340 
para_0 = torch.randn([1, 736, 7, 7], dtype=torch.float16)
para_1 = torch.randn([736], dtype=torch.float16)
para_2 = torch.randn([736], dtype=torch.float16)
para_3 = torch.randn([736], dtype=torch.float16)
para_4 = torch.randn([736], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 341 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 342 
para_0 = torch.randn([1, 736, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 343 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 344 
para_0 = torch.randn([1, 736, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 736, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 345 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 346 
para_0 = torch.randn([1, 768, 7, 7], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 347 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 348 
para_0 = torch.randn([1, 768, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 349 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 350 
para_0 = torch.randn([1, 768, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 351 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 352 
para_0 = torch.randn([1, 800, 7, 7], dtype=torch.float16)
para_1 = torch.randn([800], dtype=torch.float16)
para_2 = torch.randn([800], dtype=torch.float16)
para_3 = torch.randn([800], dtype=torch.float16)
para_4 = torch.randn([800], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 353 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 354 
para_0 = torch.randn([1, 800, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 355 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 356 
para_0 = torch.randn([1, 800, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 800, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 357 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 358 
para_0 = torch.randn([1, 832, 7, 7], dtype=torch.float16)
para_1 = torch.randn([832], dtype=torch.float16)
para_2 = torch.randn([832], dtype=torch.float16)
para_3 = torch.randn([832], dtype=torch.float16)
para_4 = torch.randn([832], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 359 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 360 
para_0 = torch.randn([1, 832, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 361 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([1024, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 59 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 60 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 362 
para_0 = torch.randn([1, 832, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 832, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 363 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 364 
para_0 = torch.randn([1, 864, 7, 7], dtype=torch.float16)
para_1 = torch.randn([864], dtype=torch.float16)
para_2 = torch.randn([864], dtype=torch.float16)
para_3 = torch.randn([864], dtype=torch.float16)
para_4 = torch.randn([864], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 365 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 366 
para_0 = torch.randn([1, 864, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 367 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 61 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 62 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 368 
para_0 = torch.randn([1, 864, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 864, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 369 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 370 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
para_1 = torch.randn([896], dtype=torch.float16)
para_2 = torch.randn([896], dtype=torch.float16)
para_3 = torch.randn([896], dtype=torch.float16)
para_4 = torch.randn([896], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 371 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 372 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 373 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 374 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 896, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 63 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 375 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 376 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
para_1 = torch.randn([928], dtype=torch.float16)
para_2 = torch.randn([928], dtype=torch.float16)
para_3 = torch.randn([928], dtype=torch.float16)
para_4 = torch.randn([928], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 377 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 378 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 379 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 380 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 928, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 381 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 382 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 383 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 384 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 385 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 386 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 387 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 388 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
para_1 = torch.randn([992], dtype=torch.float16)
para_2 = torch.randn([992], dtype=torch.float16)
para_3 = torch.randn([992], dtype=torch.float16)
para_4 = torch.randn([992], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 389 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 390 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 391 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 392 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 992, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 393 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 394 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 395 
verify_model(torch.nn.BatchNorm2d(1024,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 396 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 20 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 22 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 65 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 66 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 67 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 68 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 397 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 69 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 70 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 71 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 72 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 73 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 74 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 96, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 75 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2048, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 76 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 77 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2048], dtype=torch.float16)
para_2 = torch.randn([2048], dtype=torch.float16)
para_3 = torch.randn([2048], dtype=torch.float16)
para_4 = torch.randn([2048], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 78 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 398 
para_0 = torch.randn([1, 1024], dtype=torch.float16)
para_1 = torch.randn([1000, 1024], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 399 
verify_model(torch.nn.Linear(1024,1000,).eval(), input_data=[torch.randn([1, 1024], dtype=torch.float16)])
# test_id: 79 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([2048, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 80 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 2048, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 37 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 39 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
para_1 = torch.randn([224], dtype=torch.float16)
para_2 = torch.randn([224], dtype=torch.float16)
para_3 = torch.randn([224], dtype=torch.float16)
para_4 = torch.randn([224], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 40 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 41 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 85 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 86 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 224, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 45 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 46 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 47 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 48 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 49 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 50 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 51 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 52 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 53 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 54 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 55 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 56 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 57 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 58 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 59 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 60 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 61 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 62 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 63 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 87 
para_0 = torch.randn([1, 2048], dtype=torch.float16)
para_1 = torch.randn([1000, 2048], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 88 
verify_model(torch.nn.Linear(2048,1000,).eval(), input_data=[torch.randn([1, 2048], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 65 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 66 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 67 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 68 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 69 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 70 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 71 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 72 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
para_1 = torch.randn([224], dtype=torch.float16)
para_2 = torch.randn([224], dtype=torch.float16)
para_3 = torch.randn([224], dtype=torch.float16)
para_4 = torch.randn([224], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 73 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 74 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 75 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 76 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 224, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 77 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 78 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 79 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 80 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 85 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 86 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 87 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 88 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 89 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 90 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 91 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 92 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 93 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 94 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 95 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 96 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
para_1 = torch.randn([352], dtype=torch.float16)
para_2 = torch.randn([352], dtype=torch.float16)
para_3 = torch.randn([352], dtype=torch.float16)
para_4 = torch.randn([352], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 97 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 98 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 99 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 100 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 352, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 101 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 102 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 103 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 104 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 105 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 106 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 107 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 108 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
para_1 = torch.randn([416], dtype=torch.float16)
para_2 = torch.randn([416], dtype=torch.float16)
para_3 = torch.randn([416], dtype=torch.float16)
para_4 = torch.randn([416], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 109 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 110 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 111 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 112 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 416, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 113 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 114 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
para_1 = torch.randn([448], dtype=torch.float16)
para_2 = torch.randn([448], dtype=torch.float16)
para_3 = torch.randn([448], dtype=torch.float16)
para_4 = torch.randn([448], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 115 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 116 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 117 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 118 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 448, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 119 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 120 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 121 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 122 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 123 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 124 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 125 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 126 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 127 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 128 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 129 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 130 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 131 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 132 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 133 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 134 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 135 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 136 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 137 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 138 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 139 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 140 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 141 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 142 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 143 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 144 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 145 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 146 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 147 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 148 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 149 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 150 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 151 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 152 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 153 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 154 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 155 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 156 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 157 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 158 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
para_1 = torch.randn([352], dtype=torch.float16)
para_2 = torch.randn([352], dtype=torch.float16)
para_3 = torch.randn([352], dtype=torch.float16)
para_4 = torch.randn([352], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 159 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 160 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 161 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 162 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 352, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 163 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 164 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 165 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 166 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 167 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 168 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 169 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 170 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
para_1 = torch.randn([416], dtype=torch.float16)
para_2 = torch.randn([416], dtype=torch.float16)
para_3 = torch.randn([416], dtype=torch.float16)
para_4 = torch.randn([416], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 171 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 172 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 173 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 174 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 416, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 175 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 176 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
para_1 = torch.randn([448], dtype=torch.float16)
para_2 = torch.randn([448], dtype=torch.float16)
para_3 = torch.randn([448], dtype=torch.float16)
para_4 = torch.randn([448], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 177 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 178 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 179 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 180 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 448, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 181 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 182 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 183 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 184 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 185 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 186 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 187 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 188 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 189 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 190 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 191 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 192 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 193 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 194 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
para_1 = torch.randn([544], dtype=torch.float16)
para_2 = torch.randn([544], dtype=torch.float16)
para_3 = torch.randn([544], dtype=torch.float16)
para_4 = torch.randn([544], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 195 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 196 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 197 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 198 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 544, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 199 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 200 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 201 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 202 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 203 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 204 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 205 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 206 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
para_1 = torch.randn([608], dtype=torch.float16)
para_2 = torch.randn([608], dtype=torch.float16)
para_3 = torch.randn([608], dtype=torch.float16)
para_4 = torch.randn([608], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 207 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 208 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 209 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 96, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 210 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 608, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 211 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 212 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = torch.randn([640], dtype=torch.float16)
para_2 = torch.randn([640], dtype=torch.float16)
para_3 = torch.randn([640], dtype=torch.float16)
para_4 = torch.randn([640], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 213 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 214 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 215 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 216 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 640, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 217 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 218 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 219 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 220 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 14 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 221 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 160, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 222 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 223 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 15 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 33 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 224 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
para_1 = torch.randn([704], dtype=torch.float16)
para_2 = torch.randn([704], dtype=torch.float16)
para_3 = torch.randn([704], dtype=torch.float16)
para_4 = torch.randn([704], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 225 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 16 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 226 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 227 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 17 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 228 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 704, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 229 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 230 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
para_1 = torch.randn([736], dtype=torch.float16)
para_2 = torch.randn([736], dtype=torch.float16)
para_3 = torch.randn([736], dtype=torch.float16)
para_4 = torch.randn([736], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 231 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 232 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 233 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 234 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 736, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 235 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 236 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 237 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 238 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 239 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 39 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
para_1 = torch.randn([224], dtype=torch.float16)
para_2 = torch.randn([224], dtype=torch.float16)
para_3 = torch.randn([224], dtype=torch.float16)
para_4 = torch.randn([224], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 40 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 41 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 240 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 241 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 242 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
para_1 = torch.randn([800], dtype=torch.float16)
para_2 = torch.randn([800], dtype=torch.float16)
para_3 = torch.randn([800], dtype=torch.float16)
para_4 = torch.randn([800], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 243 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 244 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 245 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 224, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 224, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 45 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 46 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 47 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 48 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 246 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 800, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 247 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 49 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 50 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 248 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
para_1 = torch.randn([832], dtype=torch.float16)
para_2 = torch.randn([832], dtype=torch.float16)
para_3 = torch.randn([832], dtype=torch.float16)
para_4 = torch.randn([832], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 249 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 250 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 251 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 51 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 52 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 53 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 54 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 55 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 56 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 252 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 832, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 253 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 254 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([864], dtype=torch.float16)
para_2 = torch.randn([864], dtype=torch.float16)
para_3 = torch.randn([864], dtype=torch.float16)
para_4 = torch.randn([864], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 255 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 256 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 257 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 57 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 58 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 23 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 25 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 258 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 864, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 259 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 260 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = torch.randn([896], dtype=torch.float16)
para_2 = torch.randn([896], dtype=torch.float16)
para_3 = torch.randn([896], dtype=torch.float16)
para_4 = torch.randn([896], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 261 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 262 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 263 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 59 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 60 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 61 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 62 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 63 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 264 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 896, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 265 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 266 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
para_1 = torch.randn([928], dtype=torch.float16)
para_2 = torch.randn([928], dtype=torch.float16)
para_3 = torch.randn([928], dtype=torch.float16)
para_4 = torch.randn([928], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 267 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 268 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 269 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 160, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 65 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 66 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 67 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 68 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 69 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 270 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 928, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 271 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 272 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 273 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 274 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 275 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 70 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 71 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 72 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
para_1 = torch.randn([224], dtype=torch.float16)
para_2 = torch.randn([224], dtype=torch.float16)
para_3 = torch.randn([224], dtype=torch.float16)
para_4 = torch.randn([224], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 73 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 74 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 75 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 276 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 277 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([512, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 278 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
para_1 = torch.randn([992], dtype=torch.float16)
para_2 = torch.randn([992], dtype=torch.float16)
para_3 = torch.randn([992], dtype=torch.float16)
para_4 = torch.randn([992], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 279 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 280 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 281 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 76 
para_0 = torch.randn([1, 224, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 224, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 77 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 78 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 79 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 80 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 282 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 992, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 283 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 284 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 285 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 286 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 287 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 85 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 86 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 87 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 288 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 289 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 290 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1056], dtype=torch.float16)
para_2 = torch.randn([1056], dtype=torch.float16)
para_3 = torch.randn([1056], dtype=torch.float16)
para_4 = torch.randn([1056], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 291 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 292 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 293 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 88 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 89 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 90 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 91 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 92 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 93 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 294 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1056, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 295 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 296 
para_0 = torch.randn([1, 1088, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1088], dtype=torch.float16)
para_2 = torch.randn([1088], dtype=torch.float16)
para_3 = torch.randn([1088], dtype=torch.float16)
para_4 = torch.randn([1088], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 297 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 298 
para_0 = torch.randn([1, 1088, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 299 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 94 
para_0 = torch.randn([1, 320, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 95 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 96 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
para_1 = torch.randn([352], dtype=torch.float16)
para_2 = torch.randn([352], dtype=torch.float16)
para_3 = torch.randn([352], dtype=torch.float16)
para_4 = torch.randn([352], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 97 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 98 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 99 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 300 
para_0 = torch.randn([1, 1088, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1088, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 301 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 302 
para_0 = torch.randn([1, 1120, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1120], dtype=torch.float16)
para_2 = torch.randn([1120], dtype=torch.float16)
para_3 = torch.randn([1120], dtype=torch.float16)
para_4 = torch.randn([1120], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 303 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 304 
para_0 = torch.randn([1, 1120, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 305 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 100 
para_0 = torch.randn([1, 352, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 352, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 101 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 102 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 103 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 104 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 105 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 306 
para_0 = torch.randn([1, 1120, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1120, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 307 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 308 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1152], dtype=torch.float16)
para_2 = torch.randn([1152], dtype=torch.float16)
para_3 = torch.randn([1152], dtype=torch.float16)
para_4 = torch.randn([1152], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 309 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 310 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 311 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 106 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 107 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 108 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
para_1 = torch.randn([416], dtype=torch.float16)
para_2 = torch.randn([416], dtype=torch.float16)
para_3 = torch.randn([416], dtype=torch.float16)
para_4 = torch.randn([416], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 109 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 110 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 111 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 312 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1152, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 313 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 314 
para_0 = torch.randn([1, 1184, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1184], dtype=torch.float16)
para_2 = torch.randn([1184], dtype=torch.float16)
para_3 = torch.randn([1184], dtype=torch.float16)
para_4 = torch.randn([1184], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 315 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 316 
para_0 = torch.randn([1, 1184, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 317 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 112 
para_0 = torch.randn([1, 416, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 416, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 113 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 114 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
para_1 = torch.randn([448], dtype=torch.float16)
para_2 = torch.randn([448], dtype=torch.float16)
para_3 = torch.randn([448], dtype=torch.float16)
para_4 = torch.randn([448], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 115 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([1024, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 116 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 117 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 59 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 60 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 318 
para_0 = torch.randn([1, 1184, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1184, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 319 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 320 
para_0 = torch.randn([1, 1216, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1216], dtype=torch.float16)
para_2 = torch.randn([1216], dtype=torch.float16)
para_3 = torch.randn([1216], dtype=torch.float16)
para_4 = torch.randn([1216], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 321 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 322 
para_0 = torch.randn([1, 1216, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 323 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 61 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 62 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 118 
para_0 = torch.randn([1, 448, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 448, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 119 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 120 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 121 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 122 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 123 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 324 
para_0 = torch.randn([1, 1216, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1216, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 325 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 326 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1248], dtype=torch.float16)
para_2 = torch.randn([1248], dtype=torch.float16)
para_3 = torch.randn([1248], dtype=torch.float16)
para_4 = torch.randn([1248], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 327 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 328 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 329 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 124 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 125 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 63 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 126 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 127 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 128 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 129 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 330 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1248, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 331 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 332 
para_0 = torch.randn([1, 1280, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1280], dtype=torch.float16)
para_2 = torch.randn([1280], dtype=torch.float16)
para_3 = torch.randn([1280], dtype=torch.float16)
para_4 = torch.randn([1280], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 333 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 15 
para_0 = torch.randn([1, 64, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 334 
para_0 = torch.randn([1, 1280, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 335 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 130 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 131 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 132 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 133 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 134 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 135 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 136 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 137 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 336 
para_0 = torch.randn([1, 1280, 14, 14], dtype=torch.float16)
para_1 = torch.randn([640, 1280, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 337 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 338 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 339 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 340 
para_0 = torch.randn([1, 640, 7, 7], dtype=torch.float16)
para_1 = torch.randn([640], dtype=torch.float16)
para_2 = torch.randn([640], dtype=torch.float16)
para_3 = torch.randn([640], dtype=torch.float16)
para_4 = torch.randn([640], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 341 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 342 
para_0 = torch.randn([1, 640, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 343 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 138 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 139 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 140 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 141 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 142 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 143 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 144 
para_0 = torch.randn([1, 128, 14, 14], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 145 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 146 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 147 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 148 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 149 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 344 
para_0 = torch.randn([1, 640, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 640, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 345 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 346 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 347 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 348 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 349 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 150 
para_0 = torch.randn([1, 288, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 151 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 152 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 153 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 154 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 155 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 350 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 351 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 352 
para_0 = torch.randn([1, 672, 7, 7], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 353 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 354 
para_0 = torch.randn([1, 672, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 355 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 156 
para_0 = torch.randn([1, 320, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 157 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 158 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
para_1 = torch.randn([352], dtype=torch.float16)
para_2 = torch.randn([352], dtype=torch.float16)
para_3 = torch.randn([352], dtype=torch.float16)
para_4 = torch.randn([352], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 159 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 160 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 161 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 356 
para_0 = torch.randn([1, 672, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 357 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 162 
para_0 = torch.randn([1, 352, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 352, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 358 
para_0 = torch.randn([1, 704, 7, 7], dtype=torch.float16)
para_1 = torch.randn([704], dtype=torch.float16)
para_2 = torch.randn([704], dtype=torch.float16)
para_3 = torch.randn([704], dtype=torch.float16)
para_4 = torch.randn([704], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 163 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 359 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 360 
para_0 = torch.randn([1, 704, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 361 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 164 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 165 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 166 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 167 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 362 
para_0 = torch.randn([1, 704, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 704, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 363 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 168 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 169 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 364 
para_0 = torch.randn([1, 736, 7, 7], dtype=torch.float16)
para_1 = torch.randn([736], dtype=torch.float16)
para_2 = torch.randn([736], dtype=torch.float16)
para_3 = torch.randn([736], dtype=torch.float16)
para_4 = torch.randn([736], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 365 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 366 
para_0 = torch.randn([1, 736, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 367 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 170 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
para_1 = torch.randn([416], dtype=torch.float16)
para_2 = torch.randn([416], dtype=torch.float16)
para_3 = torch.randn([416], dtype=torch.float16)
para_4 = torch.randn([416], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 171 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 172 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 173 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([64, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 368 
para_0 = torch.randn([1, 736, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 736, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 369 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 370 
para_0 = torch.randn([1, 768, 7, 7], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 371 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 174 
para_0 = torch.randn([1, 416, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 416, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 175 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 372 
para_0 = torch.randn([1, 768, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 373 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 176 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
para_1 = torch.randn([448], dtype=torch.float16)
para_2 = torch.randn([448], dtype=torch.float16)
para_3 = torch.randn([448], dtype=torch.float16)
para_4 = torch.randn([448], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 177 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 178 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 179 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 374 
para_0 = torch.randn([1, 768, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 375 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 180 
para_0 = torch.randn([1, 448, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 448, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 181 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 376 
para_0 = torch.randn([1, 800, 7, 7], dtype=torch.float16)
para_1 = torch.randn([800], dtype=torch.float16)
para_2 = torch.randn([800], dtype=torch.float16)
para_3 = torch.randn([800], dtype=torch.float16)
para_4 = torch.randn([800], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 377 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 378 
para_0 = torch.randn([1, 800, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 379 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 182 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 183 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 184 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 185 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 65 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 66 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 67 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 68 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 380 
para_0 = torch.randn([1, 800, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 800, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 381 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 186 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 187 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 382 
para_0 = torch.randn([1, 832, 7, 7], dtype=torch.float16)
para_1 = torch.randn([832], dtype=torch.float16)
para_2 = torch.randn([832], dtype=torch.float16)
para_3 = torch.randn([832], dtype=torch.float16)
para_4 = torch.randn([832], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 383 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 384 
para_0 = torch.randn([1, 832, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 188 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 385 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 189 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 190 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 191 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 69 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 70 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 71 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 72 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 73 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 74 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 23 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 25 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 26 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 192 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 193 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 386 
para_0 = torch.randn([1, 832, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 832, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 387 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 194 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
para_1 = torch.randn([544], dtype=torch.float16)
para_2 = torch.randn([544], dtype=torch.float16)
para_3 = torch.randn([544], dtype=torch.float16)
para_4 = torch.randn([544], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 195 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 388 
para_0 = torch.randn([1, 864, 7, 7], dtype=torch.float16)
para_1 = torch.randn([864], dtype=torch.float16)
para_2 = torch.randn([864], dtype=torch.float16)
para_3 = torch.randn([864], dtype=torch.float16)
para_4 = torch.randn([864], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 196 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 389 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 197 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 390 
para_0 = torch.randn([1, 864, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 391 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 27 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 198 
para_0 = torch.randn([1, 544, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 544, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 199 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 392 
para_0 = torch.randn([1, 864, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 864, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 393 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 394 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
para_1 = torch.randn([896], dtype=torch.float16)
para_2 = torch.randn([896], dtype=torch.float16)
para_3 = torch.randn([896], dtype=torch.float16)
para_4 = torch.randn([896], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 395 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 396 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 397 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 200 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 201 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 202 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 203 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 75 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2048, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 76 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 77 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2048], dtype=torch.float16)
para_2 = torch.randn([2048], dtype=torch.float16)
para_3 = torch.randn([2048], dtype=torch.float16)
para_4 = torch.randn([2048], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 78 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 398 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 896, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 399 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 128, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 204 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 205 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 400 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
para_1 = torch.randn([928], dtype=torch.float16)
para_2 = torch.randn([928], dtype=torch.float16)
para_3 = torch.randn([928], dtype=torch.float16)
para_4 = torch.randn([928], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 401 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 402 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 206 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
para_1 = torch.randn([608], dtype=torch.float16)
para_2 = torch.randn([608], dtype=torch.float16)
para_3 = torch.randn([608], dtype=torch.float16)
para_4 = torch.randn([608], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 403 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 207 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 208 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 209 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 404 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 928, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 405 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 210 
para_0 = torch.randn([1, 608, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 608, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 211 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([512, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 79 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([2048, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 406 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 212 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = torch.randn([640], dtype=torch.float16)
para_2 = torch.randn([640], dtype=torch.float16)
para_3 = torch.randn([640], dtype=torch.float16)
para_4 = torch.randn([640], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 407 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 213 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 39 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 408 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 80 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 409 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 214 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 215 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 410 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 216 
para_0 = torch.randn([1, 640, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 640, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 411 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 217 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 2048, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 412 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
para_1 = torch.randn([992], dtype=torch.float16)
para_2 = torch.randn([992], dtype=torch.float16)
para_3 = torch.randn([992], dtype=torch.float16)
para_4 = torch.randn([992], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 218 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 413 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 219 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 414 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 415 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 220 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 221 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 416 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 992, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 42 
para_0 = torch.randn([1, 128, 28, 28], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 222 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 417 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 223 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 418 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 419 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 420 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 421 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 224 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
para_1 = torch.randn([704], dtype=torch.float16)
para_2 = torch.randn([704], dtype=torch.float16)
para_3 = torch.randn([704], dtype=torch.float16)
para_4 = torch.randn([704], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 225 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 226 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 227 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 422 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 423 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 228 
para_0 = torch.randn([1, 704, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 704, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 229 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 424 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1056], dtype=torch.float16)
para_2 = torch.randn([1056], dtype=torch.float16)
para_3 = torch.randn([1056], dtype=torch.float16)
para_4 = torch.randn([1056], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 425 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 426 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 427 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 230 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
para_1 = torch.randn([736], dtype=torch.float16)
para_2 = torch.randn([736], dtype=torch.float16)
para_3 = torch.randn([736], dtype=torch.float16)
para_4 = torch.randn([736], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 231 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 232 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 233 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 85 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 86 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 428 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1056, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 429 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 430 
para_0 = torch.randn([1, 1088, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1088], dtype=torch.float16)
para_2 = torch.randn([1088], dtype=torch.float16)
para_3 = torch.randn([1088], dtype=torch.float16)
para_4 = torch.randn([1088], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 431 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 432 
para_0 = torch.randn([1, 1088, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 433 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 434 
para_0 = torch.randn([1, 1088, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1088, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 435 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 436 
para_0 = torch.randn([1, 1120, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1120], dtype=torch.float16)
para_2 = torch.randn([1120], dtype=torch.float16)
para_3 = torch.randn([1120], dtype=torch.float16)
para_4 = torch.randn([1120], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 437 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 438 
para_0 = torch.randn([1, 1120, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 439 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 440 
para_0 = torch.randn([1, 1120, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1120, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 441 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 442 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1152], dtype=torch.float16)
para_2 = torch.randn([1152], dtype=torch.float16)
para_3 = torch.randn([1152], dtype=torch.float16)
para_4 = torch.randn([1152], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 443 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 444 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 445 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 234 
para_0 = torch.randn([1, 736, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 736, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 235 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024, 256, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 236 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 237 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 238 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 239 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 446 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1152, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 447 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 448 
para_0 = torch.randn([1, 1184, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1184], dtype=torch.float16)
para_2 = torch.randn([1184], dtype=torch.float16)
para_3 = torch.randn([1184], dtype=torch.float16)
para_4 = torch.randn([1184], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 449 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 450 
para_0 = torch.randn([1, 1184, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 451 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 87 
para_0 = torch.randn([1, 2048], dtype=torch.float16)
para_1 = torch.randn([1000, 2048], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 88 
verify_model(torch.nn.Linear(2048,1000,).eval(), input_data=[torch.randn([1, 2048], dtype=torch.float16)])
# test_id: 240 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 241 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([1024, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 242 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
para_1 = torch.randn([800], dtype=torch.float16)
para_2 = torch.randn([800], dtype=torch.float16)
para_3 = torch.randn([800], dtype=torch.float16)
para_4 = torch.randn([800], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 243 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 244 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 245 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 59 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 60 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 452 
para_0 = torch.randn([1, 1184, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1184, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 453 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 454 
para_0 = torch.randn([1, 1216, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1216], dtype=torch.float16)
para_2 = torch.randn([1216], dtype=torch.float16)
para_3 = torch.randn([1216], dtype=torch.float16)
para_4 = torch.randn([1216], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 455 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 456 
para_0 = torch.randn([1, 1216, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 457 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 61 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 62 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 246 
para_0 = torch.randn([1, 800, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 800, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 458 
para_0 = torch.randn([1, 1216, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1216, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 247 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 459 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 460 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1248], dtype=torch.float16)
para_2 = torch.randn([1248], dtype=torch.float16)
para_3 = torch.randn([1248], dtype=torch.float16)
para_4 = torch.randn([1248], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 248 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
para_1 = torch.randn([832], dtype=torch.float16)
para_2 = torch.randn([832], dtype=torch.float16)
para_3 = torch.randn([832], dtype=torch.float16)
para_4 = torch.randn([832], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 461 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 249 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 462 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 250 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 463 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 251 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 464 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1248, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 465 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 252 
para_0 = torch.randn([1, 832, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 832, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 253 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 466 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1280], dtype=torch.float16)
para_2 = torch.randn([1280], dtype=torch.float16)
para_3 = torch.randn([1280], dtype=torch.float16)
para_4 = torch.randn([1280], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 467 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 468 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 469 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 254 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([864], dtype=torch.float16)
para_2 = torch.randn([864], dtype=torch.float16)
para_3 = torch.randn([864], dtype=torch.float16)
para_4 = torch.randn([864], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 255 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 256 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 257 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 63 
para_0 = torch.randn([1, 256, 14, 14], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 470 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1280, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 471 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 472 
para_0 = torch.randn([1, 1312, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1312], dtype=torch.float16)
para_2 = torch.randn([1312], dtype=torch.float16)
para_3 = torch.randn([1312], dtype=torch.float16)
para_4 = torch.randn([1312], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 473 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 258 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 864, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 474 
para_0 = torch.randn([1, 1312, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 259 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 475 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 260 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = torch.randn([896], dtype=torch.float16)
para_2 = torch.randn([896], dtype=torch.float16)
para_3 = torch.randn([896], dtype=torch.float16)
para_4 = torch.randn([896], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 261 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 262 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 263 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 476 
para_0 = torch.randn([1, 1312, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1312, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 477 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 478 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1344], dtype=torch.float16)
para_2 = torch.randn([1344], dtype=torch.float16)
para_3 = torch.randn([1344], dtype=torch.float16)
para_4 = torch.randn([1344], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 479 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 480 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 481 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 264 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 896, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 265 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 266 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
para_1 = torch.randn([928], dtype=torch.float16)
para_2 = torch.randn([928], dtype=torch.float16)
para_3 = torch.randn([928], dtype=torch.float16)
para_4 = torch.randn([928], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 267 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 268 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 269 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 482 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1344, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 483 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 484 
para_0 = torch.randn([1, 1376, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1376], dtype=torch.float16)
para_2 = torch.randn([1376], dtype=torch.float16)
para_3 = torch.randn([1376], dtype=torch.float16)
para_4 = torch.randn([1376], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 485 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 486 
para_0 = torch.randn([1, 1376, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 487 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 270 
para_0 = torch.randn([1, 928, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 928, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 271 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 272 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 273 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 274 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 275 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 488 
para_0 = torch.randn([1, 1376, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1376, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 489 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 490 
para_0 = torch.randn([1, 1408, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1408], dtype=torch.float16)
para_2 = torch.randn([1408], dtype=torch.float16)
para_3 = torch.randn([1408], dtype=torch.float16)
para_4 = torch.randn([1408], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 491 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 492 
para_0 = torch.randn([1, 1408, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 493 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 276 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 277 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 278 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
para_1 = torch.randn([992], dtype=torch.float16)
para_2 = torch.randn([992], dtype=torch.float16)
para_3 = torch.randn([992], dtype=torch.float16)
para_4 = torch.randn([992], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 279 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 280 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 281 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 494 
para_0 = torch.randn([1, 1408, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1408, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 495 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 496 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1440], dtype=torch.float16)
para_2 = torch.randn([1440], dtype=torch.float16)
para_3 = torch.randn([1440], dtype=torch.float16)
para_4 = torch.randn([1440], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 497 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 498 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 499 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 282 
para_0 = torch.randn([1, 992, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 992, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 283 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 284 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 285 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 286 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 287 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 500 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1440, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 501 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 502 
para_0 = torch.randn([1, 1472, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1472], dtype=torch.float16)
para_2 = torch.randn([1472], dtype=torch.float16)
para_3 = torch.randn([1472], dtype=torch.float16)
para_4 = torch.randn([1472], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 503 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 504 
para_0 = torch.randn([1, 1472, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 505 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 288 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 289 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 290 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1056], dtype=torch.float16)
para_2 = torch.randn([1056], dtype=torch.float16)
para_3 = torch.randn([1056], dtype=torch.float16)
para_4 = torch.randn([1056], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 291 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 292 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 293 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 506 
para_0 = torch.randn([1, 1472, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1472, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 507 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 508 
para_0 = torch.randn([1, 1504, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1504], dtype=torch.float16)
para_2 = torch.randn([1504], dtype=torch.float16)
para_3 = torch.randn([1504], dtype=torch.float16)
para_4 = torch.randn([1504], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 509 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 510 
para_0 = torch.randn([1, 1504, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 511 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 294 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1056, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 295 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 296 
para_0 = torch.randn([1, 1088, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1088], dtype=torch.float16)
para_2 = torch.randn([1088], dtype=torch.float16)
para_3 = torch.randn([1088], dtype=torch.float16)
para_4 = torch.randn([1088], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 297 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 298 
para_0 = torch.randn([1, 1088, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 299 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 512 
para_0 = torch.randn([1, 1504, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1504, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 513 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 514 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1536], dtype=torch.float16)
para_2 = torch.randn([1536], dtype=torch.float16)
para_3 = torch.randn([1536], dtype=torch.float16)
para_4 = torch.randn([1536], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 515 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 516 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 517 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 300 
para_0 = torch.randn([1, 1088, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1088, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 301 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 302 
para_0 = torch.randn([1, 1120, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1120], dtype=torch.float16)
para_2 = torch.randn([1120], dtype=torch.float16)
para_3 = torch.randn([1120], dtype=torch.float16)
para_4 = torch.randn([1120], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 303 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 304 
para_0 = torch.randn([1, 1120, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 305 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 518 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1536, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 519 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 520 
para_0 = torch.randn([1, 1568, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1568], dtype=torch.float16)
para_2 = torch.randn([1568], dtype=torch.float16)
para_3 = torch.randn([1568], dtype=torch.float16)
para_4 = torch.randn([1568], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 521 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 522 
para_0 = torch.randn([1, 1568, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 523 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 306 
para_0 = torch.randn([1, 1120, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1120, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 307 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 308 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1152], dtype=torch.float16)
para_2 = torch.randn([1152], dtype=torch.float16)
para_3 = torch.randn([1152], dtype=torch.float16)
para_4 = torch.randn([1152], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 309 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 310 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 311 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 524 
para_0 = torch.randn([1, 1568, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1568, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 525 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 526 
para_0 = torch.randn([1, 1600, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1600], dtype=torch.float16)
para_2 = torch.randn([1600], dtype=torch.float16)
para_3 = torch.randn([1600], dtype=torch.float16)
para_4 = torch.randn([1600], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 527 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 528 
para_0 = torch.randn([1, 1600, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 529 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 312 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1152, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 313 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 314 
para_0 = torch.randn([1, 1184, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1184], dtype=torch.float16)
para_2 = torch.randn([1184], dtype=torch.float16)
para_3 = torch.randn([1184], dtype=torch.float16)
para_4 = torch.randn([1184], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 315 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 316 
para_0 = torch.randn([1, 1184, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 317 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 530 
para_0 = torch.randn([1, 1600, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1600, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 531 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 532 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1632], dtype=torch.float16)
para_2 = torch.randn([1632], dtype=torch.float16)
para_3 = torch.randn([1632], dtype=torch.float16)
para_4 = torch.randn([1632], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 533 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 534 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 535 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 318 
para_0 = torch.randn([1, 1184, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1184, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 319 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 320 
para_0 = torch.randn([1, 1216, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1216], dtype=torch.float16)
para_2 = torch.randn([1216], dtype=torch.float16)
para_3 = torch.randn([1216], dtype=torch.float16)
para_4 = torch.randn([1216], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 321 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 322 
para_0 = torch.randn([1, 1216, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 323 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 536 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1632, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 537 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 538 
para_0 = torch.randn([1, 1664, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1664], dtype=torch.float16)
para_2 = torch.randn([1664], dtype=torch.float16)
para_3 = torch.randn([1664], dtype=torch.float16)
para_4 = torch.randn([1664], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 539 
verify_model(torch.nn.BatchNorm2d(1664,).eval(), input_data=[torch.randn([1, 1664, 7, 7], dtype=torch.float16)])
# test_id: 540 
para_0 = torch.randn([1, 1664, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 324 
para_0 = torch.randn([1, 1216, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1216, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 325 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 326 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1248], dtype=torch.float16)
para_2 = torch.randn([1248], dtype=torch.float16)
para_3 = torch.randn([1248], dtype=torch.float16)
para_4 = torch.randn([1248], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 327 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 328 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 329 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 65 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 66 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 67 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 68 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 330 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1248, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 331 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 541 
para_0 = torch.randn([1, 1664, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 332 
para_0 = torch.randn([1, 1280, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1280], dtype=torch.float16)
para_2 = torch.randn([1280], dtype=torch.float16)
para_3 = torch.randn([1280], dtype=torch.float16)
para_4 = torch.randn([1280], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 333 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 334 
para_0 = torch.randn([1, 1280, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 335 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 69 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 70 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 71 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 72 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 73 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 74 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 336 
para_0 = torch.randn([1, 1280, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1280, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 337 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 338 
para_0 = torch.randn([1, 1312, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1312], dtype=torch.float16)
para_2 = torch.randn([1312], dtype=torch.float16)
para_3 = torch.randn([1312], dtype=torch.float16)
para_4 = torch.randn([1312], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 339 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1312, 14, 14], dtype=torch.float16)])
# test_id: 340 
para_0 = torch.randn([1, 1312, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 341 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1312, 14, 14], dtype=torch.float16)])
# test_id: 342 
para_0 = torch.randn([1, 1312, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1312, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 343 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1312, 14, 14], dtype=torch.float16)])
# test_id: 344 
para_0 = torch.randn([1, 1344, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1344], dtype=torch.float16)
para_2 = torch.randn([1344], dtype=torch.float16)
para_3 = torch.randn([1344], dtype=torch.float16)
para_4 = torch.randn([1344], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 345 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 346 
para_0 = torch.randn([1, 1344, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 347 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 348 
para_0 = torch.randn([1, 1344, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1344, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 349 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 350 
para_0 = torch.randn([1, 1376, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1376], dtype=torch.float16)
para_2 = torch.randn([1376], dtype=torch.float16)
para_3 = torch.randn([1376], dtype=torch.float16)
para_4 = torch.randn([1376], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 351 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1376, 14, 14], dtype=torch.float16)])
# test_id: 352 
para_0 = torch.randn([1, 1376, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 353 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1376, 14, 14], dtype=torch.float16)])
# test_id: 75 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2048, 512, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 76 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 77 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2048], dtype=torch.float16)
para_2 = torch.randn([2048], dtype=torch.float16)
para_3 = torch.randn([2048], dtype=torch.float16)
para_4 = torch.randn([2048], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 78 
verify_model(torch.nn.BatchNorm2d(2048,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 542 
para_0 = torch.randn([1, 1664], dtype=torch.float16)
para_1 = torch.randn([1000, 1664], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 543 
verify_model(torch.nn.Linear(1664,1000,).eval(), input_data=[torch.randn([1, 1664], dtype=torch.float16)])
# test_id: 354 
para_0 = torch.randn([1, 1376, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1376, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 355 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1376, 14, 14], dtype=torch.float16)])
# test_id: 356 
para_0 = torch.randn([1, 1408, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1408], dtype=torch.float16)
para_2 = torch.randn([1408], dtype=torch.float16)
para_3 = torch.randn([1408], dtype=torch.float16)
para_4 = torch.randn([1408], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 357 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1408, 14, 14], dtype=torch.float16)])
# test_id: 358 
para_0 = torch.randn([1, 1408, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 359 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1408, 14, 14], dtype=torch.float16)])
# test_id: 79 
para_0 = torch.randn([1, 1024, 14, 14], dtype=torch.float16)
para_1 = torch.randn([2048, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 80 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 360 
para_0 = torch.randn([1, 1408, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1408, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 361 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1408, 14, 14], dtype=torch.float16)])
# test_id: 362 
para_0 = torch.randn([1, 1440, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1440], dtype=torch.float16)
para_2 = torch.randn([1440], dtype=torch.float16)
para_3 = torch.randn([1440], dtype=torch.float16)
para_4 = torch.randn([1440], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 363 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 364 
para_0 = torch.randn([1, 1440, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 365 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 2048, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(512,2048,kernel_size=1,stride=1,bias=False,).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 366 
para_0 = torch.randn([1, 1440, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1440, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 367 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 368 
para_0 = torch.randn([1, 1472, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1472], dtype=torch.float16)
para_2 = torch.randn([1472], dtype=torch.float16)
para_3 = torch.randn([1472], dtype=torch.float16)
para_4 = torch.randn([1472], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 369 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1472, 14, 14], dtype=torch.float16)])
# test_id: 370 
para_0 = torch.randn([1, 1472, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 371 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1472, 14, 14], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 372 
para_0 = torch.randn([1, 1472, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1472, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 373 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1472, 14, 14], dtype=torch.float16)])
# test_id: 374 
para_0 = torch.randn([1, 1504, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1504], dtype=torch.float16)
para_2 = torch.randn([1504], dtype=torch.float16)
para_3 = torch.randn([1504], dtype=torch.float16)
para_4 = torch.randn([1504], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 375 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1504, 14, 14], dtype=torch.float16)])
# test_id: 376 
para_0 = torch.randn([1, 1504, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 377 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1504, 14, 14], dtype=torch.float16)])
# test_id: 378 
para_0 = torch.randn([1, 1504, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1504, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 379 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1504, 14, 14], dtype=torch.float16)])
# test_id: 85 
para_0 = torch.randn([1, 2048, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 86 
verify_model(torch.nn.AdaptiveAvgPool2d((1, 1),).eval(), input_data=[torch.randn([1, 2048, 7, 7], dtype=torch.float16)])
# test_id: 380 
para_0 = torch.randn([1, 1536, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1536], dtype=torch.float16)
para_2 = torch.randn([1536], dtype=torch.float16)
para_3 = torch.randn([1536], dtype=torch.float16)
para_4 = torch.randn([1536], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 381 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 382 
para_0 = torch.randn([1, 1536, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 383 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 384 
para_0 = torch.randn([1, 1536, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1536, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 385 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 386 
para_0 = torch.randn([1, 1568, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1568], dtype=torch.float16)
para_2 = torch.randn([1568], dtype=torch.float16)
para_3 = torch.randn([1568], dtype=torch.float16)
para_4 = torch.randn([1568], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 387 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1568, 14, 14], dtype=torch.float16)])
# test_id: 388 
para_0 = torch.randn([1, 1568, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 389 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1568, 14, 14], dtype=torch.float16)])
# test_id: 390 
para_0 = torch.randn([1, 1568, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1568, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 391 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1568, 14, 14], dtype=torch.float16)])
# test_id: 392 
para_0 = torch.randn([1, 1600, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1600], dtype=torch.float16)
para_2 = torch.randn([1600], dtype=torch.float16)
para_3 = torch.randn([1600], dtype=torch.float16)
para_4 = torch.randn([1600], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 393 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1600, 14, 14], dtype=torch.float16)])
# test_id: 394 
para_0 = torch.randn([1, 1600, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 395 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1600, 14, 14], dtype=torch.float16)])
# test_id: 396 
para_0 = torch.randn([1, 1600, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1600, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 397 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1600, 14, 14], dtype=torch.float16)])
# test_id: 398 
para_0 = torch.randn([1, 1632, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1632], dtype=torch.float16)
para_2 = torch.randn([1632], dtype=torch.float16)
para_3 = torch.randn([1632], dtype=torch.float16)
para_4 = torch.randn([1632], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 399 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 400 
para_0 = torch.randn([1, 1632, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 401 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 87 
para_0 = torch.randn([1, 2048], dtype=torch.float16)
para_1 = torch.randn([1000, 2048], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 88 
verify_model(torch.nn.Linear(2048,1000,).eval(), input_data=[torch.randn([1, 2048], dtype=torch.float16)])
# test_id: 402 
para_0 = torch.randn([1, 1632, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1632, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 403 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 404 
para_0 = torch.randn([1, 1664, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1664], dtype=torch.float16)
para_2 = torch.randn([1664], dtype=torch.float16)
para_3 = torch.randn([1664], dtype=torch.float16)
para_4 = torch.randn([1664], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 405 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1664, 14, 14], dtype=torch.float16)])
# test_id: 406 
para_0 = torch.randn([1, 1664, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 407 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1664, 14, 14], dtype=torch.float16)])
# test_id: 408 
para_0 = torch.randn([1, 1664, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1664, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 409 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1664, 14, 14], dtype=torch.float16)])
# test_id: 410 
para_0 = torch.randn([1, 1696, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1696], dtype=torch.float16)
para_2 = torch.randn([1696], dtype=torch.float16)
para_3 = torch.randn([1696], dtype=torch.float16)
para_4 = torch.randn([1696], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 411 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1696, 14, 14], dtype=torch.float16)])
# test_id: 412 
para_0 = torch.randn([1, 1696, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 413 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1696, 14, 14], dtype=torch.float16)])
# test_id: 414 
para_0 = torch.randn([1, 1696, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1696, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 415 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1696, 14, 14], dtype=torch.float16)])
# test_id: 416 
para_0 = torch.randn([1, 1728, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1728], dtype=torch.float16)
para_2 = torch.randn([1728], dtype=torch.float16)
para_3 = torch.randn([1728], dtype=torch.float16)
para_4 = torch.randn([1728], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 417 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 418 
para_0 = torch.randn([1, 1728, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 419 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 420 
para_0 = torch.randn([1, 1728, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1728, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 421 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 422 
para_0 = torch.randn([1, 1760, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1760], dtype=torch.float16)
para_2 = torch.randn([1760], dtype=torch.float16)
para_3 = torch.randn([1760], dtype=torch.float16)
para_4 = torch.randn([1760], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 423 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1760, 14, 14], dtype=torch.float16)])
# test_id: 424 
para_0 = torch.randn([1, 1760, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 425 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1760, 14, 14], dtype=torch.float16)])
# test_id: 426 
para_0 = torch.randn([1, 1760, 14, 14], dtype=torch.float16)
para_1 = torch.randn([128, 1760, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 427 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1760, 14, 14], dtype=torch.float16)])
# test_id: 428 
para_0 = torch.randn([1, 1792, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1792], dtype=torch.float16)
para_2 = torch.randn([1792], dtype=torch.float16)
para_3 = torch.randn([1792], dtype=torch.float16)
para_4 = torch.randn([1792], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 429 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1792, 14, 14], dtype=torch.float16)])
# test_id: 430 
para_0 = torch.randn([1, 1792, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 431 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1792, 14, 14], dtype=torch.float16)])
# test_id: 432 
para_0 = torch.randn([1, 1792, 14, 14], dtype=torch.float16)
para_1 = torch.randn([896, 1792, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 433 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1792, 14, 14], dtype=torch.float16)])
# test_id: 434 
para_0 = torch.randn([1, 896, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 435 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 436 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
para_1 = torch.randn([896], dtype=torch.float16)
para_2 = torch.randn([896], dtype=torch.float16)
para_3 = torch.randn([896], dtype=torch.float16)
para_4 = torch.randn([896], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 437 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 438 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 439 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 440 
para_0 = torch.randn([1, 896, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 896, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 441 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 442 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 443 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 444 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 445 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 446 
para_0 = torch.randn([1, 128, 7, 7], dtype=torch.float16)
para_1 = torch.randn([32, 128, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 447 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 448 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
para_1 = torch.randn([928], dtype=torch.float16)
para_2 = torch.randn([928], dtype=torch.float16)
para_3 = torch.randn([928], dtype=torch.float16)
para_4 = torch.randn([928], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 449 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 450 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 451 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 452 
para_0 = torch.randn([1, 928, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 928, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 453 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 454 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 455 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 456 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 457 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 458 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 459 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 460 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
para_1 = torch.randn([992], dtype=torch.float16)
para_2 = torch.randn([992], dtype=torch.float16)
para_3 = torch.randn([992], dtype=torch.float16)
para_4 = torch.randn([992], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 461 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 462 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 463 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 464 
para_0 = torch.randn([1, 992, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 992, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 465 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 466 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1024], dtype=torch.float16)
para_2 = torch.randn([1024], dtype=torch.float16)
para_3 = torch.randn([1024], dtype=torch.float16)
para_4 = torch.randn([1024], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 467 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 468 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 469 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 470 
para_0 = torch.randn([1, 1024, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1024, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 471 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 472 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1056], dtype=torch.float16)
para_2 = torch.randn([1056], dtype=torch.float16)
para_3 = torch.randn([1056], dtype=torch.float16)
para_4 = torch.randn([1056], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 473 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 474 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 475 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 476 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1056, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 477 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 478 
para_0 = torch.randn([1, 1088, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1088], dtype=torch.float16)
para_2 = torch.randn([1088], dtype=torch.float16)
para_3 = torch.randn([1088], dtype=torch.float16)
para_4 = torch.randn([1088], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 479 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 480 
para_0 = torch.randn([1, 1088, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 481 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 482 
para_0 = torch.randn([1, 1088, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1088, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 483 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 484 
para_0 = torch.randn([1, 1120, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1120], dtype=torch.float16)
para_2 = torch.randn([1120], dtype=torch.float16)
para_3 = torch.randn([1120], dtype=torch.float16)
para_4 = torch.randn([1120], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 485 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 486 
para_0 = torch.randn([1, 1120, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 487 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 488 
para_0 = torch.randn([1, 1120, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1120, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 489 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 490 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1152], dtype=torch.float16)
para_2 = torch.randn([1152], dtype=torch.float16)
para_3 = torch.randn([1152], dtype=torch.float16)
para_4 = torch.randn([1152], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 491 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 492 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 493 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 494 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1152, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 495 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 496 
para_0 = torch.randn([1, 1184, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1184], dtype=torch.float16)
para_2 = torch.randn([1184], dtype=torch.float16)
para_3 = torch.randn([1184], dtype=torch.float16)
para_4 = torch.randn([1184], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 497 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 498 
para_0 = torch.randn([1, 1184, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 499 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 500 
para_0 = torch.randn([1, 1184, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1184, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 501 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 502 
para_0 = torch.randn([1, 1216, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1216], dtype=torch.float16)
para_2 = torch.randn([1216], dtype=torch.float16)
para_3 = torch.randn([1216], dtype=torch.float16)
para_4 = torch.randn([1216], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 503 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 504 
para_0 = torch.randn([1, 1216, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 505 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 506 
para_0 = torch.randn([1, 1216, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1216, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 507 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 508 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1248], dtype=torch.float16)
para_2 = torch.randn([1248], dtype=torch.float16)
para_3 = torch.randn([1248], dtype=torch.float16)
para_4 = torch.randn([1248], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 509 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 510 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 511 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 512 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1248, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 513 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 514 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1280], dtype=torch.float16)
para_2 = torch.randn([1280], dtype=torch.float16)
para_3 = torch.randn([1280], dtype=torch.float16)
para_4 = torch.randn([1280], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 515 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 516 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 517 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 518 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1280, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 519 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 520 
para_0 = torch.randn([1, 1312, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1312], dtype=torch.float16)
para_2 = torch.randn([1312], dtype=torch.float16)
para_3 = torch.randn([1312], dtype=torch.float16)
para_4 = torch.randn([1312], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 521 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 522 
para_0 = torch.randn([1, 1312, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 523 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 524 
para_0 = torch.randn([1, 1312, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1312, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 525 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 526 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1344], dtype=torch.float16)
para_2 = torch.randn([1344], dtype=torch.float16)
para_3 = torch.randn([1344], dtype=torch.float16)
para_4 = torch.randn([1344], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 527 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 528 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 529 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 530 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1344, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 531 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 532 
para_0 = torch.randn([1, 1376, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1376], dtype=torch.float16)
para_2 = torch.randn([1376], dtype=torch.float16)
para_3 = torch.randn([1376], dtype=torch.float16)
para_4 = torch.randn([1376], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 533 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 534 
para_0 = torch.randn([1, 1376, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 535 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 536 
para_0 = torch.randn([1, 1376, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1376, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 537 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 538 
para_0 = torch.randn([1, 1408, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1408], dtype=torch.float16)
para_2 = torch.randn([1408], dtype=torch.float16)
para_3 = torch.randn([1408], dtype=torch.float16)
para_4 = torch.randn([1408], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 539 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 540 
para_0 = torch.randn([1, 1408, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 541 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 542 
para_0 = torch.randn([1, 1408, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1408, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 543 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 544 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1440], dtype=torch.float16)
para_2 = torch.randn([1440], dtype=torch.float16)
para_3 = torch.randn([1440], dtype=torch.float16)
para_4 = torch.randn([1440], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 545 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 546 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 547 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 548 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1440, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 549 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 550 
para_0 = torch.randn([1, 1472, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1472], dtype=torch.float16)
para_2 = torch.randn([1472], dtype=torch.float16)
para_3 = torch.randn([1472], dtype=torch.float16)
para_4 = torch.randn([1472], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 551 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 552 
para_0 = torch.randn([1, 1472, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 553 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 554 
para_0 = torch.randn([1, 1472, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1472, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 555 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 556 
para_0 = torch.randn([1, 1504, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1504], dtype=torch.float16)
para_2 = torch.randn([1504], dtype=torch.float16)
para_3 = torch.randn([1504], dtype=torch.float16)
para_4 = torch.randn([1504], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 557 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 558 
para_0 = torch.randn([1, 1504, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 559 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 560 
para_0 = torch.randn([1, 1504, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1504, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 561 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 562 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1536], dtype=torch.float16)
para_2 = torch.randn([1536], dtype=torch.float16)
para_3 = torch.randn([1536], dtype=torch.float16)
para_4 = torch.randn([1536], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 563 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 564 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 565 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 566 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1536, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 567 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 568 
para_0 = torch.randn([1, 1568, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1568], dtype=torch.float16)
para_2 = torch.randn([1568], dtype=torch.float16)
para_3 = torch.randn([1568], dtype=torch.float16)
para_4 = torch.randn([1568], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 569 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 570 
para_0 = torch.randn([1, 1568, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 571 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 572 
para_0 = torch.randn([1, 1568, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1568, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 573 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 574 
para_0 = torch.randn([1, 1600, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1600], dtype=torch.float16)
para_2 = torch.randn([1600], dtype=torch.float16)
para_3 = torch.randn([1600], dtype=torch.float16)
para_4 = torch.randn([1600], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 575 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 576 
para_0 = torch.randn([1, 1600, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 577 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 578 
para_0 = torch.randn([1, 1600, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1600, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 579 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 580 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1632], dtype=torch.float16)
para_2 = torch.randn([1632], dtype=torch.float16)
para_3 = torch.randn([1632], dtype=torch.float16)
para_4 = torch.randn([1632], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 581 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 582 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 583 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 584 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1632, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 585 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 586 
para_0 = torch.randn([1, 1664, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1664], dtype=torch.float16)
para_2 = torch.randn([1664], dtype=torch.float16)
para_3 = torch.randn([1664], dtype=torch.float16)
para_4 = torch.randn([1664], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 587 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1664, 7, 7], dtype=torch.float16)])
# test_id: 588 
para_0 = torch.randn([1, 1664, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 589 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1664, 7, 7], dtype=torch.float16)])
# test_id: 590 
para_0 = torch.randn([1, 1664, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1664, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 591 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1664, 7, 7], dtype=torch.float16)])
# test_id: 592 
para_0 = torch.randn([1, 1696, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1696], dtype=torch.float16)
para_2 = torch.randn([1696], dtype=torch.float16)
para_3 = torch.randn([1696], dtype=torch.float16)
para_4 = torch.randn([1696], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 593 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1696, 7, 7], dtype=torch.float16)])
# test_id: 594 
para_0 = torch.randn([1, 1696, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 595 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1696, 7, 7], dtype=torch.float16)])
# test_id: 596 
para_0 = torch.randn([1, 1696, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1696, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 597 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1696, 7, 7], dtype=torch.float16)])
# test_id: 598 
para_0 = torch.randn([1, 1728, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1728], dtype=torch.float16)
para_2 = torch.randn([1728], dtype=torch.float16)
para_3 = torch.randn([1728], dtype=torch.float16)
para_4 = torch.randn([1728], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 599 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 600 
para_0 = torch.randn([1, 1728, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 601 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 602 
para_0 = torch.randn([1, 1728, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1728, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 603 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 604 
para_0 = torch.randn([1, 1760, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1760], dtype=torch.float16)
para_2 = torch.randn([1760], dtype=torch.float16)
para_3 = torch.randn([1760], dtype=torch.float16)
para_4 = torch.randn([1760], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 605 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1760, 7, 7], dtype=torch.float16)])
# test_id: 606 
para_0 = torch.randn([1, 1760, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 607 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1760, 7, 7], dtype=torch.float16)])
# test_id: 608 
para_0 = torch.randn([1, 1760, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1760, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 609 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1760, 7, 7], dtype=torch.float16)])
# test_id: 610 
para_0 = torch.randn([1, 1792, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1792], dtype=torch.float16)
para_2 = torch.randn([1792], dtype=torch.float16)
para_3 = torch.randn([1792], dtype=torch.float16)
para_4 = torch.randn([1792], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 611 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1792, 7, 7], dtype=torch.float16)])
# test_id: 612 
para_0 = torch.randn([1, 1792, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 613 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1792, 7, 7], dtype=torch.float16)])
# test_id: 614 
para_0 = torch.randn([1, 1792, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1792, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 615 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1792, 7, 7], dtype=torch.float16)])
# test_id: 616 
para_0 = torch.randn([1, 1824, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1824], dtype=torch.float16)
para_2 = torch.randn([1824], dtype=torch.float16)
para_3 = torch.randn([1824], dtype=torch.float16)
para_4 = torch.randn([1824], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 617 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 618 
para_0 = torch.randn([1, 1824, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 619 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 620 
para_0 = torch.randn([1, 1824, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1824, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 621 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 622 
para_0 = torch.randn([1, 1856, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1856], dtype=torch.float16)
para_2 = torch.randn([1856], dtype=torch.float16)
para_3 = torch.randn([1856], dtype=torch.float16)
para_4 = torch.randn([1856], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 623 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1856, 7, 7], dtype=torch.float16)])
# test_id: 624 
para_0 = torch.randn([1, 1856, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 625 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1856, 7, 7], dtype=torch.float16)])
# test_id: 626 
para_0 = torch.randn([1, 1856, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1856, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 627 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1856, 7, 7], dtype=torch.float16)])
# test_id: 628 
para_0 = torch.randn([1, 1888, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1888], dtype=torch.float16)
para_2 = torch.randn([1888], dtype=torch.float16)
para_3 = torch.randn([1888], dtype=torch.float16)
para_4 = torch.randn([1888], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 629 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1888, 7, 7], dtype=torch.float16)])
# test_id: 630 
para_0 = torch.randn([1, 1888, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 631 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1888, 7, 7], dtype=torch.float16)])
# test_id: 632 
para_0 = torch.randn([1, 1888, 7, 7], dtype=torch.float16)
para_1 = torch.randn([128, 1888, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 633 
verify_model(torch.nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1888, 7, 7], dtype=torch.float16)])
# test_id: 634 
para_0 = torch.randn([1, 1920, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1920], dtype=torch.float16)
para_2 = torch.randn([1920], dtype=torch.float16)
para_3 = torch.randn([1920], dtype=torch.float16)
para_4 = torch.randn([1920], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 635 
verify_model(torch.nn.BatchNorm2d(1920,).eval(), input_data=[torch.randn([1, 1920, 7, 7], dtype=torch.float16)])
# test_id: 636 
para_0 = torch.randn([1, 1920, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 637 
para_0 = torch.randn([1, 1920, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 638 
para_0 = torch.randn([1, 1920], dtype=torch.float16)
para_1 = torch.randn([1000, 1920], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 639 
verify_model(torch.nn.Linear(1920,1000,).eval(), input_data=[torch.randn([1, 1920], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([96, 3, 7, 7], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (3, 3)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 96, 112, 112], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 96, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 96, 112, 112], dtype=torch.float16)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 96, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([48, 192, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = torch.randn([144], dtype=torch.float16)
para_2 = torch.randn([144], dtype=torch.float16)
para_3 = torch.randn([144], dtype=torch.float16)
para_4 = torch.randn([144], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 144, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
para_0 = torch.randn([1, 240, 56, 56], dtype=torch.float16)
para_1 = torch.randn([240], dtype=torch.float16)
para_2 = torch.randn([240], dtype=torch.float16)
para_3 = torch.randn([240], dtype=torch.float16)
para_4 = torch.randn([240], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 28 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 240, 56, 56], dtype=torch.float16)])
# test_id: 29 
para_0 = torch.randn([1, 240, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 30 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 240, 56, 56], dtype=torch.float16)])
# test_id: 31 
para_0 = torch.randn([1, 240, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 240, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 32 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 240, 56, 56], dtype=torch.float16)])
# test_id: 33 
para_0 = torch.randn([1, 288, 56, 56], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 34 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 288, 56, 56], dtype=torch.float16)])
# test_id: 35 
para_0 = torch.randn([1, 288, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 36 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 56, 56], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 288, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 56, 56], dtype=torch.float16)])
# test_id: 39 
para_0 = torch.randn([1, 336, 56, 56], dtype=torch.float16)
para_1 = torch.randn([336], dtype=torch.float16)
para_2 = torch.randn([336], dtype=torch.float16)
para_3 = torch.randn([336], dtype=torch.float16)
para_4 = torch.randn([336], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 40 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 336, 56, 56], dtype=torch.float16)])
# test_id: 41 
para_0 = torch.randn([1, 336, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 336, 56, 56], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 336, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 336, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 44 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 336, 56, 56], dtype=torch.float16)])
# test_id: 45 
para_0 = torch.randn([1, 384, 56, 56], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 46 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 384, 56, 56], dtype=torch.float16)])
# test_id: 47 
para_0 = torch.randn([1, 384, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 48 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 56, 56], dtype=torch.float16)])
# test_id: 49 
para_0 = torch.randn([1, 384, 56, 56], dtype=torch.float16)
para_1 = torch.randn([192, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 50 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 56, 56], dtype=torch.float16)])
# test_id: 51 
para_0 = torch.randn([1, 192, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 52 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 53 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 54 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 55 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 56 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 57 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 58 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 59 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([48, 192, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 60 
para_0 = torch.randn([1, 240, 28, 28], dtype=torch.float16)
para_1 = torch.randn([240], dtype=torch.float16)
para_2 = torch.randn([240], dtype=torch.float16)
para_3 = torch.randn([240], dtype=torch.float16)
para_4 = torch.randn([240], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 61 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 240, 28, 28], dtype=torch.float16)])
# test_id: 62 
para_0 = torch.randn([1, 240, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 63 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 240, 28, 28], dtype=torch.float16)])
# test_id: 64 
para_0 = torch.randn([1, 240, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 240, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 65 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 240, 28, 28], dtype=torch.float16)])
# test_id: 66 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([288], dtype=torch.float16)
para_2 = torch.randn([288], dtype=torch.float16)
para_3 = torch.randn([288], dtype=torch.float16)
para_4 = torch.randn([288], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 67 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 68 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 69 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 70 
para_0 = torch.randn([1, 288, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 288, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 71 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 72 
para_0 = torch.randn([1, 336, 28, 28], dtype=torch.float16)
para_1 = torch.randn([336], dtype=torch.float16)
para_2 = torch.randn([336], dtype=torch.float16)
para_3 = torch.randn([336], dtype=torch.float16)
para_4 = torch.randn([336], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 73 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 336, 28, 28], dtype=torch.float16)])
# test_id: 74 
para_0 = torch.randn([1, 336, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 75 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 336, 28, 28], dtype=torch.float16)])
# test_id: 76 
para_0 = torch.randn([1, 336, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 336, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 77 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 336, 28, 28], dtype=torch.float16)])
# test_id: 78 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 79 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 80 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 81 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 82 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 83 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 84 
para_0 = torch.randn([1, 432, 28, 28], dtype=torch.float16)
para_1 = torch.randn([432], dtype=torch.float16)
para_2 = torch.randn([432], dtype=torch.float16)
para_3 = torch.randn([432], dtype=torch.float16)
para_4 = torch.randn([432], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 85 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 432, 28, 28], dtype=torch.float16)])
# test_id: 86 
para_0 = torch.randn([1, 432, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 87 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 432, 28, 28], dtype=torch.float16)])
# test_id: 88 
para_0 = torch.randn([1, 432, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 432, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 89 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 432, 28, 28], dtype=torch.float16)])
# test_id: 90 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 91 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 92 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 93 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 94 
para_0 = torch.randn([1, 480, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 95 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 96 
para_0 = torch.randn([1, 528, 28, 28], dtype=torch.float16)
para_1 = torch.randn([528], dtype=torch.float16)
para_2 = torch.randn([528], dtype=torch.float16)
para_3 = torch.randn([528], dtype=torch.float16)
para_4 = torch.randn([528], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 97 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 528, 28, 28], dtype=torch.float16)])
# test_id: 98 
para_0 = torch.randn([1, 528, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 99 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 528, 28, 28], dtype=torch.float16)])
# test_id: 100 
para_0 = torch.randn([1, 528, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 528, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 101 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 528, 28, 28], dtype=torch.float16)])
# test_id: 102 
para_0 = torch.randn([1, 576, 28, 28], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 103 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 576, 28, 28], dtype=torch.float16)])
# test_id: 104 
para_0 = torch.randn([1, 576, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 105 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 576, 28, 28], dtype=torch.float16)])
# test_id: 106 
para_0 = torch.randn([1, 576, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 107 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 28, 28], dtype=torch.float16)])
# test_id: 108 
para_0 = torch.randn([1, 624, 28, 28], dtype=torch.float16)
para_1 = torch.randn([624], dtype=torch.float16)
para_2 = torch.randn([624], dtype=torch.float16)
para_3 = torch.randn([624], dtype=torch.float16)
para_4 = torch.randn([624], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 109 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 624, 28, 28], dtype=torch.float16)])
# test_id: 110 
para_0 = torch.randn([1, 624, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 111 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 624, 28, 28], dtype=torch.float16)])
# test_id: 112 
para_0 = torch.randn([1, 624, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 624, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 113 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 624, 28, 28], dtype=torch.float16)])
# test_id: 114 
para_0 = torch.randn([1, 672, 28, 28], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 115 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 672, 28, 28], dtype=torch.float16)])
# test_id: 116 
para_0 = torch.randn([1, 672, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 117 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 28, 28], dtype=torch.float16)])
# test_id: 118 
para_0 = torch.randn([1, 672, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 119 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 28, 28], dtype=torch.float16)])
# test_id: 120 
para_0 = torch.randn([1, 720, 28, 28], dtype=torch.float16)
para_1 = torch.randn([720], dtype=torch.float16)
para_2 = torch.randn([720], dtype=torch.float16)
para_3 = torch.randn([720], dtype=torch.float16)
para_4 = torch.randn([720], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 121 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 720, 28, 28], dtype=torch.float16)])
# test_id: 122 
para_0 = torch.randn([1, 720, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 123 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 720, 28, 28], dtype=torch.float16)])
# test_id: 124 
para_0 = torch.randn([1, 720, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 720, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 125 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 720, 28, 28], dtype=torch.float16)])
# test_id: 126 
para_0 = torch.randn([1, 768, 28, 28], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 127 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 768, 28, 28], dtype=torch.float16)])
# test_id: 128 
para_0 = torch.randn([1, 768, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 129 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 28, 28], dtype=torch.float16)])
# test_id: 130 
para_0 = torch.randn([1, 768, 28, 28], dtype=torch.float16)
para_1 = torch.randn([384, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 131 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 28, 28], dtype=torch.float16)])
# test_id: 132 
para_0 = torch.randn([1, 384, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 133 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 134 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 135 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 136 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 137 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 138 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 139 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 140 
para_0 = torch.randn([1, 192, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 141 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 142 
para_0 = torch.randn([1, 192, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 143 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 144 
para_0 = torch.randn([1, 192, 14, 14], dtype=torch.float16)
para_1 = torch.randn([48, 192, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 145 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 146 
para_0 = torch.randn([1, 432, 14, 14], dtype=torch.float16)
para_1 = torch.randn([432], dtype=torch.float16)
para_2 = torch.randn([432], dtype=torch.float16)
para_3 = torch.randn([432], dtype=torch.float16)
para_4 = torch.randn([432], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 147 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 432, 14, 14], dtype=torch.float16)])
# test_id: 148 
para_0 = torch.randn([1, 432, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 149 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 432, 14, 14], dtype=torch.float16)])
# test_id: 150 
para_0 = torch.randn([1, 432, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 432, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 151 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 432, 14, 14], dtype=torch.float16)])
# test_id: 152 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([480], dtype=torch.float16)
para_2 = torch.randn([480], dtype=torch.float16)
para_3 = torch.randn([480], dtype=torch.float16)
para_4 = torch.randn([480], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 153 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 154 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 155 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 156 
para_0 = torch.randn([1, 480, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 480, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 157 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 158 
para_0 = torch.randn([1, 528, 14, 14], dtype=torch.float16)
para_1 = torch.randn([528], dtype=torch.float16)
para_2 = torch.randn([528], dtype=torch.float16)
para_3 = torch.randn([528], dtype=torch.float16)
para_4 = torch.randn([528], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 159 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 528, 14, 14], dtype=torch.float16)])
# test_id: 160 
para_0 = torch.randn([1, 528, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 161 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 528, 14, 14], dtype=torch.float16)])
# test_id: 162 
para_0 = torch.randn([1, 528, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 528, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 163 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 528, 14, 14], dtype=torch.float16)])
# test_id: 164 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 165 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 166 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 167 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 168 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 169 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 170 
para_0 = torch.randn([1, 624, 14, 14], dtype=torch.float16)
para_1 = torch.randn([624], dtype=torch.float16)
para_2 = torch.randn([624], dtype=torch.float16)
para_3 = torch.randn([624], dtype=torch.float16)
para_4 = torch.randn([624], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 171 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 624, 14, 14], dtype=torch.float16)])
# test_id: 172 
para_0 = torch.randn([1, 624, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 173 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 624, 14, 14], dtype=torch.float16)])
# test_id: 174 
para_0 = torch.randn([1, 624, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 624, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 175 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 624, 14, 14], dtype=torch.float16)])
# test_id: 176 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([672], dtype=torch.float16)
para_2 = torch.randn([672], dtype=torch.float16)
para_3 = torch.randn([672], dtype=torch.float16)
para_4 = torch.randn([672], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 177 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 178 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 179 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 180 
para_0 = torch.randn([1, 672, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 672, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 181 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 182 
para_0 = torch.randn([1, 720, 14, 14], dtype=torch.float16)
para_1 = torch.randn([720], dtype=torch.float16)
para_2 = torch.randn([720], dtype=torch.float16)
para_3 = torch.randn([720], dtype=torch.float16)
para_4 = torch.randn([720], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 183 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 720, 14, 14], dtype=torch.float16)])
# test_id: 184 
para_0 = torch.randn([1, 720, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 185 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 720, 14, 14], dtype=torch.float16)])
# test_id: 186 
para_0 = torch.randn([1, 720, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 720, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 187 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 720, 14, 14], dtype=torch.float16)])
# test_id: 188 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([768], dtype=torch.float16)
para_2 = torch.randn([768], dtype=torch.float16)
para_3 = torch.randn([768], dtype=torch.float16)
para_4 = torch.randn([768], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 189 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 190 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 191 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 192 
para_0 = torch.randn([1, 768, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 768, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 193 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 194 
para_0 = torch.randn([1, 816, 14, 14], dtype=torch.float16)
para_1 = torch.randn([816], dtype=torch.float16)
para_2 = torch.randn([816], dtype=torch.float16)
para_3 = torch.randn([816], dtype=torch.float16)
para_4 = torch.randn([816], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 195 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 816, 14, 14], dtype=torch.float16)])
# test_id: 196 
para_0 = torch.randn([1, 816, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 197 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 816, 14, 14], dtype=torch.float16)])
# test_id: 198 
para_0 = torch.randn([1, 816, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 816, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 199 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 816, 14, 14], dtype=torch.float16)])
# test_id: 200 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([864], dtype=torch.float16)
para_2 = torch.randn([864], dtype=torch.float16)
para_3 = torch.randn([864], dtype=torch.float16)
para_4 = torch.randn([864], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 201 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 202 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 203 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 204 
para_0 = torch.randn([1, 864, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 864, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 205 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 206 
para_0 = torch.randn([1, 912, 14, 14], dtype=torch.float16)
para_1 = torch.randn([912], dtype=torch.float16)
para_2 = torch.randn([912], dtype=torch.float16)
para_3 = torch.randn([912], dtype=torch.float16)
para_4 = torch.randn([912], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 207 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 912, 14, 14], dtype=torch.float16)])
# test_id: 208 
para_0 = torch.randn([1, 912, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 209 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 912, 14, 14], dtype=torch.float16)])
# test_id: 210 
para_0 = torch.randn([1, 912, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 912, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 211 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 912, 14, 14], dtype=torch.float16)])
# test_id: 212 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 213 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 214 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 215 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 216 
para_0 = torch.randn([1, 960, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 217 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 218 
para_0 = torch.randn([1, 1008, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1008], dtype=torch.float16)
para_2 = torch.randn([1008], dtype=torch.float16)
para_3 = torch.randn([1008], dtype=torch.float16)
para_4 = torch.randn([1008], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 219 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1008, 14, 14], dtype=torch.float16)])
# test_id: 220 
para_0 = torch.randn([1, 1008, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 221 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1008, 14, 14], dtype=torch.float16)])
# test_id: 222 
para_0 = torch.randn([1, 1008, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1008, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 223 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1008, 14, 14], dtype=torch.float16)])
# test_id: 224 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1056], dtype=torch.float16)
para_2 = torch.randn([1056], dtype=torch.float16)
para_3 = torch.randn([1056], dtype=torch.float16)
para_4 = torch.randn([1056], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 225 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 226 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 227 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 228 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1056, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 229 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 230 
para_0 = torch.randn([1, 1104, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1104], dtype=torch.float16)
para_2 = torch.randn([1104], dtype=torch.float16)
para_3 = torch.randn([1104], dtype=torch.float16)
para_4 = torch.randn([1104], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 231 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1104, 14, 14], dtype=torch.float16)])
# test_id: 232 
para_0 = torch.randn([1, 1104, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 233 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1104, 14, 14], dtype=torch.float16)])
# test_id: 234 
para_0 = torch.randn([1, 1104, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1104, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 235 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1104, 14, 14], dtype=torch.float16)])
# test_id: 236 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1152], dtype=torch.float16)
para_2 = torch.randn([1152], dtype=torch.float16)
para_3 = torch.randn([1152], dtype=torch.float16)
para_4 = torch.randn([1152], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 237 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 238 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 239 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 240 
para_0 = torch.randn([1, 1152, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1152, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 241 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 242 
para_0 = torch.randn([1, 1200, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1200], dtype=torch.float16)
para_2 = torch.randn([1200], dtype=torch.float16)
para_3 = torch.randn([1200], dtype=torch.float16)
para_4 = torch.randn([1200], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 243 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1200, 14, 14], dtype=torch.float16)])
# test_id: 244 
para_0 = torch.randn([1, 1200, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 245 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1200, 14, 14], dtype=torch.float16)])
# test_id: 246 
para_0 = torch.randn([1, 1200, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1200, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 247 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1200, 14, 14], dtype=torch.float16)])
# test_id: 248 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1248], dtype=torch.float16)
para_2 = torch.randn([1248], dtype=torch.float16)
para_3 = torch.randn([1248], dtype=torch.float16)
para_4 = torch.randn([1248], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 249 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 250 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 251 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 252 
para_0 = torch.randn([1, 1248, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1248, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 253 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 254 
para_0 = torch.randn([1, 1296, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1296], dtype=torch.float16)
para_2 = torch.randn([1296], dtype=torch.float16)
para_3 = torch.randn([1296], dtype=torch.float16)
para_4 = torch.randn([1296], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 255 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1296, 14, 14], dtype=torch.float16)])
# test_id: 256 
para_0 = torch.randn([1, 1296, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 257 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1296, 14, 14], dtype=torch.float16)])
# test_id: 258 
para_0 = torch.randn([1, 1296, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1296, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 259 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1296, 14, 14], dtype=torch.float16)])
# test_id: 260 
para_0 = torch.randn([1, 1344, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1344], dtype=torch.float16)
para_2 = torch.randn([1344], dtype=torch.float16)
para_3 = torch.randn([1344], dtype=torch.float16)
para_4 = torch.randn([1344], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 261 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 262 
para_0 = torch.randn([1, 1344, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 263 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 264 
para_0 = torch.randn([1, 1344, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1344, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 265 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 266 
para_0 = torch.randn([1, 1392, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1392], dtype=torch.float16)
para_2 = torch.randn([1392], dtype=torch.float16)
para_3 = torch.randn([1392], dtype=torch.float16)
para_4 = torch.randn([1392], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 267 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1392, 14, 14], dtype=torch.float16)])
# test_id: 268 
para_0 = torch.randn([1, 1392, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 269 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1392, 14, 14], dtype=torch.float16)])
# test_id: 270 
para_0 = torch.randn([1, 1392, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1392, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 271 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1392, 14, 14], dtype=torch.float16)])
# test_id: 272 
para_0 = torch.randn([1, 1440, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1440], dtype=torch.float16)
para_2 = torch.randn([1440], dtype=torch.float16)
para_3 = torch.randn([1440], dtype=torch.float16)
para_4 = torch.randn([1440], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 273 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 274 
para_0 = torch.randn([1, 1440, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 275 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 276 
para_0 = torch.randn([1, 1440, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1440, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 277 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 278 
para_0 = torch.randn([1, 1488, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1488], dtype=torch.float16)
para_2 = torch.randn([1488], dtype=torch.float16)
para_3 = torch.randn([1488], dtype=torch.float16)
para_4 = torch.randn([1488], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 279 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1488, 14, 14], dtype=torch.float16)])
# test_id: 280 
para_0 = torch.randn([1, 1488, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 281 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1488, 14, 14], dtype=torch.float16)])
# test_id: 282 
para_0 = torch.randn([1, 1488, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1488, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 283 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1488, 14, 14], dtype=torch.float16)])
# test_id: 284 
para_0 = torch.randn([1, 1536, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1536], dtype=torch.float16)
para_2 = torch.randn([1536], dtype=torch.float16)
para_3 = torch.randn([1536], dtype=torch.float16)
para_4 = torch.randn([1536], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 285 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 286 
para_0 = torch.randn([1, 1536, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 287 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 288 
para_0 = torch.randn([1, 1536, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1536, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 289 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 290 
para_0 = torch.randn([1, 1584, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1584], dtype=torch.float16)
para_2 = torch.randn([1584], dtype=torch.float16)
para_3 = torch.randn([1584], dtype=torch.float16)
para_4 = torch.randn([1584], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 291 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1584, 14, 14], dtype=torch.float16)])
# test_id: 292 
para_0 = torch.randn([1, 1584, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 293 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1584, 14, 14], dtype=torch.float16)])
# test_id: 294 
para_0 = torch.randn([1, 1584, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1584, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 295 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1584, 14, 14], dtype=torch.float16)])
# test_id: 296 
para_0 = torch.randn([1, 1632, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1632], dtype=torch.float16)
para_2 = torch.randn([1632], dtype=torch.float16)
para_3 = torch.randn([1632], dtype=torch.float16)
para_4 = torch.randn([1632], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 297 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 298 
para_0 = torch.randn([1, 1632, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 299 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 300 
para_0 = torch.randn([1, 1632, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1632, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 301 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 302 
para_0 = torch.randn([1, 1680, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1680], dtype=torch.float16)
para_2 = torch.randn([1680], dtype=torch.float16)
para_3 = torch.randn([1680], dtype=torch.float16)
para_4 = torch.randn([1680], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 303 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1680, 14, 14], dtype=torch.float16)])
# test_id: 304 
para_0 = torch.randn([1, 1680, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 305 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1680, 14, 14], dtype=torch.float16)])
# test_id: 306 
para_0 = torch.randn([1, 1680, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1680, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 307 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1680, 14, 14], dtype=torch.float16)])
# test_id: 308 
para_0 = torch.randn([1, 1728, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1728], dtype=torch.float16)
para_2 = torch.randn([1728], dtype=torch.float16)
para_3 = torch.randn([1728], dtype=torch.float16)
para_4 = torch.randn([1728], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 309 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 310 
para_0 = torch.randn([1, 1728, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 311 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 312 
para_0 = torch.randn([1, 1728, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1728, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 313 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 314 
para_0 = torch.randn([1, 1776, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1776], dtype=torch.float16)
para_2 = torch.randn([1776], dtype=torch.float16)
para_3 = torch.randn([1776], dtype=torch.float16)
para_4 = torch.randn([1776], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 315 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1776, 14, 14], dtype=torch.float16)])
# test_id: 316 
para_0 = torch.randn([1, 1776, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 317 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1776, 14, 14], dtype=torch.float16)])
# test_id: 318 
para_0 = torch.randn([1, 1776, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1776, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 319 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1776, 14, 14], dtype=torch.float16)])
# test_id: 320 
para_0 = torch.randn([1, 1824, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1824], dtype=torch.float16)
para_2 = torch.randn([1824], dtype=torch.float16)
para_3 = torch.randn([1824], dtype=torch.float16)
para_4 = torch.randn([1824], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 321 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1824, 14, 14], dtype=torch.float16)])
# test_id: 322 
para_0 = torch.randn([1, 1824, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 323 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1824, 14, 14], dtype=torch.float16)])
# test_id: 324 
para_0 = torch.randn([1, 1824, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1824, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 325 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1824, 14, 14], dtype=torch.float16)])
# test_id: 326 
para_0 = torch.randn([1, 1872, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1872], dtype=torch.float16)
para_2 = torch.randn([1872], dtype=torch.float16)
para_3 = torch.randn([1872], dtype=torch.float16)
para_4 = torch.randn([1872], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 327 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1872, 14, 14], dtype=torch.float16)])
# test_id: 328 
para_0 = torch.randn([1, 1872, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 329 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1872, 14, 14], dtype=torch.float16)])
# test_id: 330 
para_0 = torch.randn([1, 1872, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1872, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 331 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1872, 14, 14], dtype=torch.float16)])
# test_id: 332 
para_0 = torch.randn([1, 1920, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1920], dtype=torch.float16)
para_2 = torch.randn([1920], dtype=torch.float16)
para_3 = torch.randn([1920], dtype=torch.float16)
para_4 = torch.randn([1920], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 333 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1920, 14, 14], dtype=torch.float16)])
# test_id: 334 
para_0 = torch.randn([1, 1920, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 335 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1920, 14, 14], dtype=torch.float16)])
# test_id: 336 
para_0 = torch.randn([1, 1920, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1920, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 337 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1920, 14, 14], dtype=torch.float16)])
# test_id: 338 
para_0 = torch.randn([1, 1968, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1968], dtype=torch.float16)
para_2 = torch.randn([1968], dtype=torch.float16)
para_3 = torch.randn([1968], dtype=torch.float16)
para_4 = torch.randn([1968], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 339 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1968, 14, 14], dtype=torch.float16)])
# test_id: 340 
para_0 = torch.randn([1, 1968, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 341 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1968, 14, 14], dtype=torch.float16)])
# test_id: 342 
para_0 = torch.randn([1, 1968, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 1968, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 343 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1968, 14, 14], dtype=torch.float16)])
# test_id: 344 
para_0 = torch.randn([1, 2016, 14, 14], dtype=torch.float16)
para_1 = torch.randn([2016], dtype=torch.float16)
para_2 = torch.randn([2016], dtype=torch.float16)
para_3 = torch.randn([2016], dtype=torch.float16)
para_4 = torch.randn([2016], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 345 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2016, 14, 14], dtype=torch.float16)])
# test_id: 346 
para_0 = torch.randn([1, 2016, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 347 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2016, 14, 14], dtype=torch.float16)])
# test_id: 348 
para_0 = torch.randn([1, 2016, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 2016, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 349 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2016, 14, 14], dtype=torch.float16)])
# test_id: 350 
para_0 = torch.randn([1, 2064, 14, 14], dtype=torch.float16)
para_1 = torch.randn([2064], dtype=torch.float16)
para_2 = torch.randn([2064], dtype=torch.float16)
para_3 = torch.randn([2064], dtype=torch.float16)
para_4 = torch.randn([2064], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 351 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2064, 14, 14], dtype=torch.float16)])
# test_id: 352 
para_0 = torch.randn([1, 2064, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 353 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2064, 14, 14], dtype=torch.float16)])
# test_id: 354 
para_0 = torch.randn([1, 2064, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192, 2064, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 355 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2064, 14, 14], dtype=torch.float16)])
# test_id: 356 
para_0 = torch.randn([1, 2112, 14, 14], dtype=torch.float16)
para_1 = torch.randn([2112], dtype=torch.float16)
para_2 = torch.randn([2112], dtype=torch.float16)
para_3 = torch.randn([2112], dtype=torch.float16)
para_4 = torch.randn([2112], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 357 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2112, 14, 14], dtype=torch.float16)])
# test_id: 358 
para_0 = torch.randn([1, 2112, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 359 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2112, 14, 14], dtype=torch.float16)])
# test_id: 360 
para_0 = torch.randn([1, 2112, 14, 14], dtype=torch.float16)
para_1 = torch.randn([1056, 2112, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 361 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2112, 14, 14], dtype=torch.float16)])
# test_id: 362 
para_0 = torch.randn([1, 1056, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 363 
verify_model(torch.nn.AvgPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 364 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1056], dtype=torch.float16)
para_2 = torch.randn([1056], dtype=torch.float16)
para_3 = torch.randn([1056], dtype=torch.float16)
para_4 = torch.randn([1056], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 365 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 366 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 367 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 368 
para_0 = torch.randn([1, 1056, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1056, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 369 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 370 
para_0 = torch.randn([1, 192, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 371 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 192, 7, 7], dtype=torch.float16)])
# test_id: 372 
para_0 = torch.randn([1, 192, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 373 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 192, 7, 7], dtype=torch.float16)])
# test_id: 374 
para_0 = torch.randn([1, 192, 7, 7], dtype=torch.float16)
para_1 = torch.randn([48, 192, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 375 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 7, 7], dtype=torch.float16)])
# test_id: 376 
para_0 = torch.randn([1, 1104, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1104], dtype=torch.float16)
para_2 = torch.randn([1104], dtype=torch.float16)
para_3 = torch.randn([1104], dtype=torch.float16)
para_4 = torch.randn([1104], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 377 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1104, 7, 7], dtype=torch.float16)])
# test_id: 378 
para_0 = torch.randn([1, 1104, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 379 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1104, 7, 7], dtype=torch.float16)])
# test_id: 380 
para_0 = torch.randn([1, 1104, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1104, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 381 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1104, 7, 7], dtype=torch.float16)])
# test_id: 382 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1152], dtype=torch.float16)
para_2 = torch.randn([1152], dtype=torch.float16)
para_3 = torch.randn([1152], dtype=torch.float16)
para_4 = torch.randn([1152], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 383 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 384 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 385 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 386 
para_0 = torch.randn([1, 1152, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1152, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 387 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 388 
para_0 = torch.randn([1, 1200, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1200], dtype=torch.float16)
para_2 = torch.randn([1200], dtype=torch.float16)
para_3 = torch.randn([1200], dtype=torch.float16)
para_4 = torch.randn([1200], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 389 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1200, 7, 7], dtype=torch.float16)])
# test_id: 390 
para_0 = torch.randn([1, 1200, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 391 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1200, 7, 7], dtype=torch.float16)])
# test_id: 392 
para_0 = torch.randn([1, 1200, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1200, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 393 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1200, 7, 7], dtype=torch.float16)])
# test_id: 394 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1248], dtype=torch.float16)
para_2 = torch.randn([1248], dtype=torch.float16)
para_3 = torch.randn([1248], dtype=torch.float16)
para_4 = torch.randn([1248], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 395 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 396 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 397 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 398 
para_0 = torch.randn([1, 1248, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1248, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 399 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 400 
para_0 = torch.randn([1, 1296, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1296], dtype=torch.float16)
para_2 = torch.randn([1296], dtype=torch.float16)
para_3 = torch.randn([1296], dtype=torch.float16)
para_4 = torch.randn([1296], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 401 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1296, 7, 7], dtype=torch.float16)])
# test_id: 402 
para_0 = torch.randn([1, 1296, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 403 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1296, 7, 7], dtype=torch.float16)])
# test_id: 404 
para_0 = torch.randn([1, 1296, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1296, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 405 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1296, 7, 7], dtype=torch.float16)])
# test_id: 406 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1344], dtype=torch.float16)
para_2 = torch.randn([1344], dtype=torch.float16)
para_3 = torch.randn([1344], dtype=torch.float16)
para_4 = torch.randn([1344], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 407 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 408 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 409 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 410 
para_0 = torch.randn([1, 1344, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1344, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 411 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 412 
para_0 = torch.randn([1, 1392, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1392], dtype=torch.float16)
para_2 = torch.randn([1392], dtype=torch.float16)
para_3 = torch.randn([1392], dtype=torch.float16)
para_4 = torch.randn([1392], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 413 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1392, 7, 7], dtype=torch.float16)])
# test_id: 414 
para_0 = torch.randn([1, 1392, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 415 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1392, 7, 7], dtype=torch.float16)])
# test_id: 416 
para_0 = torch.randn([1, 1392, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1392, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 417 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1392, 7, 7], dtype=torch.float16)])
# test_id: 418 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1440], dtype=torch.float16)
para_2 = torch.randn([1440], dtype=torch.float16)
para_3 = torch.randn([1440], dtype=torch.float16)
para_4 = torch.randn([1440], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 419 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 420 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 421 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 422 
para_0 = torch.randn([1, 1440, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1440, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 423 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 424 
para_0 = torch.randn([1, 1488, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1488], dtype=torch.float16)
para_2 = torch.randn([1488], dtype=torch.float16)
para_3 = torch.randn([1488], dtype=torch.float16)
para_4 = torch.randn([1488], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 425 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1488, 7, 7], dtype=torch.float16)])
# test_id: 426 
para_0 = torch.randn([1, 1488, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 427 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1488, 7, 7], dtype=torch.float16)])
# test_id: 428 
para_0 = torch.randn([1, 1488, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1488, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 429 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1488, 7, 7], dtype=torch.float16)])
# test_id: 430 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1536], dtype=torch.float16)
para_2 = torch.randn([1536], dtype=torch.float16)
para_3 = torch.randn([1536], dtype=torch.float16)
para_4 = torch.randn([1536], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 431 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 432 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 433 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 434 
para_0 = torch.randn([1, 1536, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1536, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 435 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 436 
para_0 = torch.randn([1, 1584, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1584], dtype=torch.float16)
para_2 = torch.randn([1584], dtype=torch.float16)
para_3 = torch.randn([1584], dtype=torch.float16)
para_4 = torch.randn([1584], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 437 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1584, 7, 7], dtype=torch.float16)])
# test_id: 438 
para_0 = torch.randn([1, 1584, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 439 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1584, 7, 7], dtype=torch.float16)])
# test_id: 440 
para_0 = torch.randn([1, 1584, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1584, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 441 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1584, 7, 7], dtype=torch.float16)])
# test_id: 442 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1632], dtype=torch.float16)
para_2 = torch.randn([1632], dtype=torch.float16)
para_3 = torch.randn([1632], dtype=torch.float16)
para_4 = torch.randn([1632], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 443 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 444 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 445 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 446 
para_0 = torch.randn([1, 1632, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1632, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 447 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 448 
para_0 = torch.randn([1, 1680, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1680], dtype=torch.float16)
para_2 = torch.randn([1680], dtype=torch.float16)
para_3 = torch.randn([1680], dtype=torch.float16)
para_4 = torch.randn([1680], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 449 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1680, 7, 7], dtype=torch.float16)])
# test_id: 450 
para_0 = torch.randn([1, 1680, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 451 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1680, 7, 7], dtype=torch.float16)])
# test_id: 452 
para_0 = torch.randn([1, 1680, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1680, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 453 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1680, 7, 7], dtype=torch.float16)])
# test_id: 454 
para_0 = torch.randn([1, 1728, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1728], dtype=torch.float16)
para_2 = torch.randn([1728], dtype=torch.float16)
para_3 = torch.randn([1728], dtype=torch.float16)
para_4 = torch.randn([1728], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 455 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 456 
para_0 = torch.randn([1, 1728, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 457 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 458 
para_0 = torch.randn([1, 1728, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1728, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 459 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 460 
para_0 = torch.randn([1, 1776, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1776], dtype=torch.float16)
para_2 = torch.randn([1776], dtype=torch.float16)
para_3 = torch.randn([1776], dtype=torch.float16)
para_4 = torch.randn([1776], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 461 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1776, 7, 7], dtype=torch.float16)])
# test_id: 462 
para_0 = torch.randn([1, 1776, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 463 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1776, 7, 7], dtype=torch.float16)])
# test_id: 464 
para_0 = torch.randn([1, 1776, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1776, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 465 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1776, 7, 7], dtype=torch.float16)])
# test_id: 466 
para_0 = torch.randn([1, 1824, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1824], dtype=torch.float16)
para_2 = torch.randn([1824], dtype=torch.float16)
para_3 = torch.randn([1824], dtype=torch.float16)
para_4 = torch.randn([1824], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 467 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 468 
para_0 = torch.randn([1, 1824, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 469 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 470 
para_0 = torch.randn([1, 1824, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1824, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 471 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 472 
para_0 = torch.randn([1, 1872, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1872], dtype=torch.float16)
para_2 = torch.randn([1872], dtype=torch.float16)
para_3 = torch.randn([1872], dtype=torch.float16)
para_4 = torch.randn([1872], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 473 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1872, 7, 7], dtype=torch.float16)])
# test_id: 474 
para_0 = torch.randn([1, 1872, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 475 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1872, 7, 7], dtype=torch.float16)])
# test_id: 476 
para_0 = torch.randn([1, 1872, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1872, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 477 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1872, 7, 7], dtype=torch.float16)])
# test_id: 478 
para_0 = torch.randn([1, 1920, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1920], dtype=torch.float16)
para_2 = torch.randn([1920], dtype=torch.float16)
para_3 = torch.randn([1920], dtype=torch.float16)
para_4 = torch.randn([1920], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 479 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1920, 7, 7], dtype=torch.float16)])
# test_id: 480 
para_0 = torch.randn([1, 1920, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 481 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1920, 7, 7], dtype=torch.float16)])
# test_id: 482 
para_0 = torch.randn([1, 1920, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1920, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 483 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1920, 7, 7], dtype=torch.float16)])
# test_id: 484 
para_0 = torch.randn([1, 1968, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1968], dtype=torch.float16)
para_2 = torch.randn([1968], dtype=torch.float16)
para_3 = torch.randn([1968], dtype=torch.float16)
para_4 = torch.randn([1968], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 485 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 1968, 7, 7], dtype=torch.float16)])
# test_id: 486 
para_0 = torch.randn([1, 1968, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 487 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 1968, 7, 7], dtype=torch.float16)])
# test_id: 488 
para_0 = torch.randn([1, 1968, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 1968, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 489 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 1968, 7, 7], dtype=torch.float16)])
# test_id: 490 
para_0 = torch.randn([1, 2016, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2016], dtype=torch.float16)
para_2 = torch.randn([2016], dtype=torch.float16)
para_3 = torch.randn([2016], dtype=torch.float16)
para_4 = torch.randn([2016], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 491 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2016, 7, 7], dtype=torch.float16)])
# test_id: 492 
para_0 = torch.randn([1, 2016, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 493 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2016, 7, 7], dtype=torch.float16)])
# test_id: 494 
para_0 = torch.randn([1, 2016, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 2016, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 495 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2016, 7, 7], dtype=torch.float16)])
# test_id: 496 
para_0 = torch.randn([1, 2064, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2064], dtype=torch.float16)
para_2 = torch.randn([2064], dtype=torch.float16)
para_3 = torch.randn([2064], dtype=torch.float16)
para_4 = torch.randn([2064], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 497 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2064, 7, 7], dtype=torch.float16)])
# test_id: 498 
para_0 = torch.randn([1, 2064, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 499 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2064, 7, 7], dtype=torch.float16)])
# test_id: 500 
para_0 = torch.randn([1, 2064, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 2064, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 501 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2064, 7, 7], dtype=torch.float16)])
# test_id: 502 
para_0 = torch.randn([1, 2112, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2112], dtype=torch.float16)
para_2 = torch.randn([2112], dtype=torch.float16)
para_3 = torch.randn([2112], dtype=torch.float16)
para_4 = torch.randn([2112], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 503 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2112, 7, 7], dtype=torch.float16)])
# test_id: 504 
para_0 = torch.randn([1, 2112, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 505 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2112, 7, 7], dtype=torch.float16)])
# test_id: 506 
para_0 = torch.randn([1, 2112, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 2112, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 507 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2112, 7, 7], dtype=torch.float16)])
# test_id: 508 
para_0 = torch.randn([1, 2160, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2160], dtype=torch.float16)
para_2 = torch.randn([2160], dtype=torch.float16)
para_3 = torch.randn([2160], dtype=torch.float16)
para_4 = torch.randn([2160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 509 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2160, 7, 7], dtype=torch.float16)])
# test_id: 510 
para_0 = torch.randn([1, 2160, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 511 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([1, 2160, 7, 7], dtype=torch.float16)])
# test_id: 512 
para_0 = torch.randn([1, 2160, 7, 7], dtype=torch.float16)
para_1 = torch.randn([192, 2160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 513 
verify_model(torch.nn.Conv2d(192,48,kernel_size=3,stride=1,padding=1,bias=False,).eval(), input_data=[torch.randn([1, 2160, 7, 7], dtype=torch.float16)])
# test_id: 514 
para_0 = torch.randn([1, 2208, 7, 7], dtype=torch.float16)
para_1 = torch.randn([2208], dtype=torch.float16)
para_2 = torch.randn([2208], dtype=torch.float16)
para_3 = torch.randn([2208], dtype=torch.float16)
para_4 = torch.randn([2208], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 515 
verify_model(torch.nn.BatchNorm2d(2208,).eval(), input_data=[torch.randn([1, 2208, 7, 7], dtype=torch.float16)])
# test_id: 516 
para_0 = torch.randn([1, 2208, 7, 7], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 517 
para_0 = torch.randn([1, 2208, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 518 
para_0 = torch.randn([1, 2208], dtype=torch.float16)
para_1 = torch.randn([1000, 2208], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 519 
verify_model(torch.nn.Linear(2208,1000,).eval(), input_data=[torch.randn([1, 2208], dtype=torch.float16)])
# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 3, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 6 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = torch.randn([64, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 7 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 8 
para_0 = torch.randn([1, 64, 224, 224], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 9 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 64, 224, 224], dtype=torch.float16)])
# test_id: 10 
para_0 = torch.randn([1, 64, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 64, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = torch.randn([128], dtype=torch.float16)
para_4 = torch.randn([128], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = torch.randn([128, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([128], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 18 
para_0 = torch.randn([1, 128, 112, 112], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 19 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 128, 112, 112], dtype=torch.float16)])
# test_id: 20 
para_0 = torch.randn([1, 128, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 128, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 21 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 22 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = torch.randn([256], dtype=torch.float16)
para_4 = torch.randn([256], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 23 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 24 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 25 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = torch.randn([256, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([256], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 256, 56, 56], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 256, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 256, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 36 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 37 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 38 
para_0 = torch.randn([1, 512, 28, 28], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 39 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 40 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512, 512, 3, 3], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
verify_model(torch.nn.Conv2d(512,512,kernel_size=3,padding=1,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 42 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = torch.randn([512], dtype=torch.float16)
para_2 = torch.randn([512], dtype=torch.float16)
para_3 = torch.randn([512], dtype=torch.float16)
para_4 = torch.randn([512], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 43 
verify_model(torch.nn.BatchNorm2d(512,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 44 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 45 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 512, 14, 14], dtype=torch.float16)
para_1 = 2
para_2 = 2
para_3 = 0
para_4 = 1
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,ceil_mode=False,return_indices=False,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.MaxPool2d(kernel_size=2,stride=2,).eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 512, 7, 7], dtype=torch.float16)
para_1 = (7, 7)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.AdaptiveAvgPool2d((7, 7),).eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 25088], dtype=torch.float16)
para_1 = torch.randn([4096, 25088], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 25088], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.ReLU(True,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = 0.5
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Dropout(p=0.5,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 56 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([4096, 4096], dtype=torch.float16)
para_2 = torch.randn([4096], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 57 
verify_model(torch.nn.Linear(4096,1000,).eval(), input_data=[torch.randn([1, 4096], dtype=torch.float16)])
# test_id: 58 
para_0 = torch.randn([1, 4096], dtype=torch.float16)
para_1 = torch.randn([1000, 4096], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 0 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float16)
para_1 = torch.randn([32, 3, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 1 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 3, 224, 224], dtype=torch.float16)])
# test_id: 2 
para_0 = torch.randn([1, 32, 112, 112], dtype=torch.float16)
para_1 = torch.randn([32], dtype=torch.float16)
para_2 = torch.randn([32], dtype=torch.float16)
para_3 = torch.randn([32], dtype=torch.float16)
para_4 = torch.randn([32], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 3 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 32, 112, 112], dtype=torch.float16)])
# test_id: 4 
para_0 = torch.randn([1, 32, 112, 112], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 5 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 32, 112, 112], dtype=torch.float16)])
# test_id: 6 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 32, 112, 112], dtype=torch.float16)])
# test_id: 7 
para_0 = torch.randn([1, 32, 112, 112], dtype=torch.float16)
para_1 = torch.randn([32, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 32
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 8 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 32, 112, 112], dtype=torch.float16)])
# test_id: 9 
para_0 = torch.randn([1, 32, 112, 112], dtype=torch.float16)
para_1 = torch.randn([16, 32, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 10 
para_0 = torch.randn([1, 16, 112, 112], dtype=torch.float16)
para_1 = torch.randn([16], dtype=torch.float16)
para_2 = torch.randn([16], dtype=torch.float16)
para_3 = torch.randn([16], dtype=torch.float16)
para_4 = torch.randn([16], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 11 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 16, 112, 112], dtype=torch.float16)])
# test_id: 12 
para_0 = torch.randn([1, 16, 112, 112], dtype=torch.float16)
para_1 = torch.randn([96, 16, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 13 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 16, 112, 112], dtype=torch.float16)])
# test_id: 14 
para_0 = torch.randn([1, 96, 112, 112], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 15 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 16 
para_0 = torch.randn([1, 96, 112, 112], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 17 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 18 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 19 
para_0 = torch.randn([1, 96, 112, 112], dtype=torch.float16)
para_1 = torch.randn([96, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 96
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 20 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 21 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 22 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 23 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 24 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 25 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 26 
para_0 = torch.randn([1, 96, 56, 56], dtype=torch.float16)
para_1 = torch.randn([24, 96, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 27 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 28 
para_0 = torch.randn([1, 24, 56, 56], dtype=torch.float16)
para_1 = torch.randn([24], dtype=torch.float16)
para_2 = torch.randn([24], dtype=torch.float16)
para_3 = torch.randn([24], dtype=torch.float16)
para_4 = torch.randn([24], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 29 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 24, 56, 56], dtype=torch.float16)])
# test_id: 30 
para_0 = torch.randn([1, 24, 56, 56], dtype=torch.float16)
para_1 = torch.randn([144, 24, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 31 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 24, 56, 56], dtype=torch.float16)])
# test_id: 32 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = torch.randn([144], dtype=torch.float16)
para_2 = torch.randn([144], dtype=torch.float16)
para_3 = torch.randn([144], dtype=torch.float16)
para_4 = torch.randn([144], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 33 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 34 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 35 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 36 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 37 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = torch.randn([144, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 144
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 38 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 39 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = torch.randn([24, 144, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 40 
para_0 = torch.randn([1, 144, 56, 56], dtype=torch.float16)
para_1 = torch.randn([144, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 144
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 41 
para_0 = torch.randn([1, 144, 28, 28], dtype=torch.float16)
para_1 = torch.randn([144], dtype=torch.float16)
para_2 = torch.randn([144], dtype=torch.float16)
para_3 = torch.randn([144], dtype=torch.float16)
para_4 = torch.randn([144], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 42 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 144, 28, 28], dtype=torch.float16)])
# test_id: 43 
para_0 = torch.randn([1, 144, 28, 28], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 44 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 144, 28, 28], dtype=torch.float16)])
# test_id: 45 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 144, 28, 28], dtype=torch.float16)])
# test_id: 46 
para_0 = torch.randn([1, 144, 28, 28], dtype=torch.float16)
para_1 = torch.randn([32, 144, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 47 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 144, 28, 28], dtype=torch.float16)])
# test_id: 48 
para_0 = torch.randn([1, 32, 28, 28], dtype=torch.float16)
para_1 = torch.randn([32], dtype=torch.float16)
para_2 = torch.randn([32], dtype=torch.float16)
para_3 = torch.randn([32], dtype=torch.float16)
para_4 = torch.randn([32], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 49 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 32, 28, 28], dtype=torch.float16)])
# test_id: 50 
para_0 = torch.randn([1, 32, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 32, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 51 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 32, 28, 28], dtype=torch.float16)])
# test_id: 52 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 53 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 54 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 55 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 56 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 57 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 192
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 58 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 59 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([32, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 60 
para_0 = torch.randn([1, 192, 28, 28], dtype=torch.float16)
para_1 = torch.randn([192, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 192
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 61 
para_0 = torch.randn([1, 192, 14, 14], dtype=torch.float16)
para_1 = torch.randn([192], dtype=torch.float16)
para_2 = torch.randn([192], dtype=torch.float16)
para_3 = torch.randn([192], dtype=torch.float16)
para_4 = torch.randn([192], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 62 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 63 
para_0 = torch.randn([1, 192, 14, 14], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 64 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 65 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 66 
para_0 = torch.randn([1, 192, 14, 14], dtype=torch.float16)
para_1 = torch.randn([64, 192, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 67 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 68 
para_0 = torch.randn([1, 64, 14, 14], dtype=torch.float16)
para_1 = torch.randn([64], dtype=torch.float16)
para_2 = torch.randn([64], dtype=torch.float16)
para_3 = torch.randn([64], dtype=torch.float16)
para_4 = torch.randn([64], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 69 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 64, 14, 14], dtype=torch.float16)])
# test_id: 70 
para_0 = torch.randn([1, 64, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384, 64, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 71 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 64, 14, 14], dtype=torch.float16)])
# test_id: 72 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384], dtype=torch.float16)
para_2 = torch.randn([384], dtype=torch.float16)
para_3 = torch.randn([384], dtype=torch.float16)
para_4 = torch.randn([384], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 73 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 74 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 75 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 76 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 77 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([384, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 384
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 78 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 79 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([64, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 80 
para_0 = torch.randn([1, 384, 14, 14], dtype=torch.float16)
para_1 = torch.randn([96, 384, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 81 
para_0 = torch.randn([1, 96, 14, 14], dtype=torch.float16)
para_1 = torch.randn([96], dtype=torch.float16)
para_2 = torch.randn([96], dtype=torch.float16)
para_3 = torch.randn([96], dtype=torch.float16)
para_4 = torch.randn([96], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 82 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 96, 14, 14], dtype=torch.float16)])
# test_id: 83 
para_0 = torch.randn([1, 96, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576, 96, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 84 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 96, 14, 14], dtype=torch.float16)])
# test_id: 85 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 86 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 87 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 88 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 89 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 90 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 576
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 91 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 92 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([96, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 93 
para_0 = torch.randn([1, 576, 14, 14], dtype=torch.float16)
para_1 = torch.randn([576, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 576
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 94 
para_0 = torch.randn([1, 576, 7, 7], dtype=torch.float16)
para_1 = torch.randn([576], dtype=torch.float16)
para_2 = torch.randn([576], dtype=torch.float16)
para_3 = torch.randn([576], dtype=torch.float16)
para_4 = torch.randn([576], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 95 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 96 
para_0 = torch.randn([1, 576, 7, 7], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 97 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 98 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 99 
para_0 = torch.randn([1, 576, 7, 7], dtype=torch.float16)
para_1 = torch.randn([160, 576, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 100 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 101 
para_0 = torch.randn([1, 160, 7, 7], dtype=torch.float16)
para_1 = torch.randn([160], dtype=torch.float16)
para_2 = torch.randn([160], dtype=torch.float16)
para_3 = torch.randn([160], dtype=torch.float16)
para_4 = torch.randn([160], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 102 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 160, 7, 7], dtype=torch.float16)])
# test_id: 103 
para_0 = torch.randn([1, 160, 7, 7], dtype=torch.float16)
para_1 = torch.randn([960, 160, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 104 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 160, 7, 7], dtype=torch.float16)])
# test_id: 105 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([960], dtype=torch.float16)
para_2 = torch.randn([960], dtype=torch.float16)
para_3 = torch.randn([960], dtype=torch.float16)
para_4 = torch.randn([960], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 106 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 107 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 108 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 109 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 110 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([960, 1, 3, 3], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 960
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 111 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 112 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([160, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 113 
para_0 = torch.randn([1, 960, 7, 7], dtype=torch.float16)
para_1 = torch.randn([320, 960, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 114 
para_0 = torch.randn([1, 320, 7, 7], dtype=torch.float16)
para_1 = torch.randn([320], dtype=torch.float16)
para_2 = torch.randn([320], dtype=torch.float16)
para_3 = torch.randn([320], dtype=torch.float16)
para_4 = torch.randn([320], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 115 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 320, 7, 7], dtype=torch.float16)])
# test_id: 116 
para_0 = torch.randn([1, 320, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1280, 320, 1, 1], dtype=torch.float16)
para_2 = None
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 117 
verify_model(torch.nn.Conv2d(320,1280,1,1,0,dilation=1,groups=1,bias=False,).eval(), input_data=[torch.randn([1, 320, 7, 7], dtype=torch.float16)])
# test_id: 118 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = torch.randn([1280], dtype=torch.float16)
para_2 = torch.randn([1280], dtype=torch.float16)
para_3 = torch.randn([1280], dtype=torch.float16)
para_4 = torch.randn([1280], dtype=torch.float16)
para_5 = False
para_6 = 0.1
para_7 = 1e-05
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 119 
verify_model(torch.nn.BatchNorm2d(1280,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 120 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 121 
verify_model(torch.nn.Hardtanh(0.0,6.0,True,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 122 
verify_model(torch.nn.ReLU6(inplace=True,).eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 123 
para_0 = torch.randn([1, 1280, 7, 7], dtype=torch.float16)
para_1 = (1, 1)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 124 
para_0 = torch.randn([1, 1280], dtype=torch.float16)
para_1 = 0.2
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 125 
verify_model(torch.nn.Dropout(p=0.2,).eval(), input_data=[torch.randn([1, 1280], dtype=torch.float16)])
# test_id: 126 
para_0 = torch.randn([1, 1280], dtype=torch.float16)
para_1 = torch.randn([1000, 1280], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 127 
verify_model(torch.nn.Linear(1280,1000,).eval(), input_data=[torch.randn([1, 1280], dtype=torch.float16)])
# test_id: 400 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 401 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 402 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 403 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 404 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 405 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 406 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 407 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 408 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 409 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 410 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 411 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 412 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 413 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 414 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 415 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 416 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 417 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 418 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 419 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 420 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 421 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 422 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 423 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 424 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 425 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 426 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 427 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 428 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 429 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 430 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 431 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 432 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 433 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 434 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 435 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 436 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 437 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 438 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 439 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 440 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 441 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 442 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 443 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 444 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 445 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 446 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 447 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 7, 7], dtype=torch.float16)])
# test_id: 448 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 449 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 544, 7, 7], dtype=torch.float16)])
# test_id: 450 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 576, 7, 7], dtype=torch.float16)])
# test_id: 451 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 608, 7, 7], dtype=torch.float16)])
# test_id: 452 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 453 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 454 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 455 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 456 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 457 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 458 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 459 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 460 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 461 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 462 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 463 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 520 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 96, 112, 112], dtype=torch.float16)])
# test_id: 521 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 522 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 523 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 144, 56, 56], dtype=torch.float16)])
# test_id: 524 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 240, 56, 56], dtype=torch.float16)])
# test_id: 525 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 56, 56], dtype=torch.float16)])
# test_id: 526 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 336, 56, 56], dtype=torch.float16)])
# test_id: 527 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 56, 56], dtype=torch.float16)])
# test_id: 528 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 529 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 240, 28, 28], dtype=torch.float16)])
# test_id: 530 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 531 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 336, 28, 28], dtype=torch.float16)])
# test_id: 532 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 533 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 432, 28, 28], dtype=torch.float16)])
# test_id: 534 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 535 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 528, 28, 28], dtype=torch.float16)])
# test_id: 536 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 576, 28, 28], dtype=torch.float16)])
# test_id: 537 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 624, 28, 28], dtype=torch.float16)])
# test_id: 538 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 28, 28], dtype=torch.float16)])
# test_id: 539 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 720, 28, 28], dtype=torch.float16)])
# test_id: 540 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 28, 28], dtype=torch.float16)])
# test_id: 541 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 542 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 14, 14], dtype=torch.float16)])
# test_id: 543 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 432, 14, 14], dtype=torch.float16)])
# test_id: 544 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 545 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 528, 14, 14], dtype=torch.float16)])
# test_id: 546 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 547 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 624, 14, 14], dtype=torch.float16)])
# test_id: 548 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 549 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 720, 14, 14], dtype=torch.float16)])
# test_id: 550 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 551 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 816, 14, 14], dtype=torch.float16)])
# test_id: 552 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 553 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 912, 14, 14], dtype=torch.float16)])
# test_id: 554 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 555 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1008, 14, 14], dtype=torch.float16)])
# test_id: 556 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 557 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1104, 14, 14], dtype=torch.float16)])
# test_id: 558 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 559 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1200, 14, 14], dtype=torch.float16)])
# test_id: 560 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 561 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1296, 14, 14], dtype=torch.float16)])
# test_id: 562 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 563 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1392, 14, 14], dtype=torch.float16)])
# test_id: 564 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 565 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1488, 14, 14], dtype=torch.float16)])
# test_id: 566 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 567 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1584, 14, 14], dtype=torch.float16)])
# test_id: 568 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 569 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1680, 14, 14], dtype=torch.float16)])
# test_id: 570 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 571 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1776, 14, 14], dtype=torch.float16)])
# test_id: 572 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1824, 14, 14], dtype=torch.float16)])
# test_id: 573 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1872, 14, 14], dtype=torch.float16)])
# test_id: 574 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1920, 14, 14], dtype=torch.float16)])
# test_id: 575 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1968, 14, 14], dtype=torch.float16)])
# test_id: 576 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2016, 14, 14], dtype=torch.float16)])
# test_id: 577 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2064, 14, 14], dtype=torch.float16)])
# test_id: 578 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2112, 14, 14], dtype=torch.float16)])
# test_id: 579 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 580 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 7, 7], dtype=torch.float16)])
# test_id: 581 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1104, 7, 7], dtype=torch.float16)])
# test_id: 582 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 583 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1200, 7, 7], dtype=torch.float16)])
# test_id: 584 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 585 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1296, 7, 7], dtype=torch.float16)])
# test_id: 586 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 587 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1392, 7, 7], dtype=torch.float16)])
# test_id: 588 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 589 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1488, 7, 7], dtype=torch.float16)])
# test_id: 590 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 591 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1584, 7, 7], dtype=torch.float16)])
# test_id: 592 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 593 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1680, 7, 7], dtype=torch.float16)])
# test_id: 594 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 595 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1776, 7, 7], dtype=torch.float16)])
# test_id: 596 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 597 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1872, 7, 7], dtype=torch.float16)])
# test_id: 598 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1920, 7, 7], dtype=torch.float16)])
# test_id: 599 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1968, 7, 7], dtype=torch.float16)])
# test_id: 600 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2016, 7, 7], dtype=torch.float16)])
# test_id: 601 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2064, 7, 7], dtype=torch.float16)])
# test_id: 602 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2112, 7, 7], dtype=torch.float16)])
# test_id: 603 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 2160, 7, 7], dtype=torch.float16)])
# test_id: 544 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 545 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 546 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 547 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 548 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 549 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 550 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 551 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 552 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 553 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 554 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 555 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 556 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 557 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 558 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 559 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 560 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 561 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 562 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 563 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 564 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 565 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 566 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 567 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 568 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 569 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 570 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 571 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 572 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 573 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 574 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 575 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 576 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 577 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 578 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 579 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 580 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 581 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 582 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 583 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 584 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 585 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 586 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 587 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 588 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 589 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 590 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 591 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 592 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 593 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 594 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 595 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 596 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 597 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 598 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 599 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 640, 7, 7], dtype=torch.float16)])
# test_id: 600 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 601 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 7, 7], dtype=torch.float16)])
# test_id: 602 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 704, 7, 7], dtype=torch.float16)])
# test_id: 603 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 736, 7, 7], dtype=torch.float16)])
# test_id: 604 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 7, 7], dtype=torch.float16)])
# test_id: 605 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 800, 7, 7], dtype=torch.float16)])
# test_id: 606 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 832, 7, 7], dtype=torch.float16)])
# test_id: 607 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 864, 7, 7], dtype=torch.float16)])
# test_id: 608 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 609 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 610 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 611 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 612 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 613 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 614 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 615 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 616 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 617 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 618 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 619 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 620 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 621 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 622 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 623 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 624 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 625 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 626 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 627 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 628 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 629 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 630 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 631 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 640 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 64, 112, 112], dtype=torch.float16)])
# test_id: 641 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 64, 56, 56], dtype=torch.float16)])
# test_id: 642 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 56, 56], dtype=torch.float16)])
# test_id: 643 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 96, 56, 56], dtype=torch.float16)])
# test_id: 644 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 160, 56, 56], dtype=torch.float16)])
# test_id: 645 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 56, 56], dtype=torch.float16)])
# test_id: 646 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 224, 56, 56], dtype=torch.float16)])
# test_id: 647 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 56, 56], dtype=torch.float16)])
# test_id: 648 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 28, 28], dtype=torch.float16)])
# test_id: 649 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 160, 28, 28], dtype=torch.float16)])
# test_id: 650 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 192, 28, 28], dtype=torch.float16)])
# test_id: 651 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 224, 28, 28], dtype=torch.float16)])
# test_id: 652 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 28, 28], dtype=torch.float16)])
# test_id: 653 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 28, 28], dtype=torch.float16)])
# test_id: 654 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 320, 28, 28], dtype=torch.float16)])
# test_id: 655 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 352, 28, 28], dtype=torch.float16)])
# test_id: 656 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 28, 28], dtype=torch.float16)])
# test_id: 657 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 416, 28, 28], dtype=torch.float16)])
# test_id: 658 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 448, 28, 28], dtype=torch.float16)])
# test_id: 659 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 28, 28], dtype=torch.float16)])
# test_id: 660 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 28, 28], dtype=torch.float16)])
# test_id: 661 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 256, 14, 14], dtype=torch.float16)])
# test_id: 662 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 14, 14], dtype=torch.float16)])
# test_id: 663 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 288, 14, 14], dtype=torch.float16)])
# test_id: 664 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 320, 14, 14], dtype=torch.float16)])
# test_id: 665 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 352, 14, 14], dtype=torch.float16)])
# test_id: 666 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 384, 14, 14], dtype=torch.float16)])
# test_id: 667 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 416, 14, 14], dtype=torch.float16)])
# test_id: 668 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 448, 14, 14], dtype=torch.float16)])
# test_id: 669 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 480, 14, 14], dtype=torch.float16)])
# test_id: 670 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 512, 14, 14], dtype=torch.float16)])
# test_id: 671 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 544, 14, 14], dtype=torch.float16)])
# test_id: 672 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 576, 14, 14], dtype=torch.float16)])
# test_id: 673 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 608, 14, 14], dtype=torch.float16)])
# test_id: 674 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 640, 14, 14], dtype=torch.float16)])
# test_id: 675 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 672, 14, 14], dtype=torch.float16)])
# test_id: 676 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 704, 14, 14], dtype=torch.float16)])
# test_id: 677 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 736, 14, 14], dtype=torch.float16)])
# test_id: 678 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 768, 14, 14], dtype=torch.float16)])
# test_id: 679 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 800, 14, 14], dtype=torch.float16)])
# test_id: 680 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 832, 14, 14], dtype=torch.float16)])
# test_id: 681 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 864, 14, 14], dtype=torch.float16)])
# test_id: 682 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 896, 14, 14], dtype=torch.float16)])
# test_id: 683 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 928, 14, 14], dtype=torch.float16)])
# test_id: 684 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 14, 14], dtype=torch.float16)])
# test_id: 685 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 992, 14, 14], dtype=torch.float16)])
# test_id: 686 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1024, 14, 14], dtype=torch.float16)])
# test_id: 687 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1056, 14, 14], dtype=torch.float16)])
# test_id: 688 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1088, 14, 14], dtype=torch.float16)])
# test_id: 689 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1120, 14, 14], dtype=torch.float16)])
# test_id: 690 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1152, 14, 14], dtype=torch.float16)])
# test_id: 691 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1184, 14, 14], dtype=torch.float16)])
# test_id: 692 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1216, 14, 14], dtype=torch.float16)])
# test_id: 693 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1248, 14, 14], dtype=torch.float16)])
# test_id: 694 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1280, 14, 14], dtype=torch.float16)])
# test_id: 695 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1312, 14, 14], dtype=torch.float16)])
# test_id: 696 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1344, 14, 14], dtype=torch.float16)])
# test_id: 697 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1376, 14, 14], dtype=torch.float16)])
# test_id: 698 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1408, 14, 14], dtype=torch.float16)])
# test_id: 699 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1440, 14, 14], dtype=torch.float16)])
# test_id: 700 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1472, 14, 14], dtype=torch.float16)])
# test_id: 701 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1504, 14, 14], dtype=torch.float16)])
# test_id: 702 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1536, 14, 14], dtype=torch.float16)])
# test_id: 703 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1568, 14, 14], dtype=torch.float16)])
# test_id: 704 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1600, 14, 14], dtype=torch.float16)])
# test_id: 705 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1632, 14, 14], dtype=torch.float16)])
# test_id: 706 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1664, 14, 14], dtype=torch.float16)])
# test_id: 707 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1696, 14, 14], dtype=torch.float16)])
# test_id: 708 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1728, 14, 14], dtype=torch.float16)])
# test_id: 709 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1760, 14, 14], dtype=torch.float16)])
# test_id: 710 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1792, 14, 14], dtype=torch.float16)])
# test_id: 711 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 896, 7, 7], dtype=torch.float16)])
# test_id: 712 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 128, 7, 7], dtype=torch.float16)])
# test_id: 713 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 928, 7, 7], dtype=torch.float16)])
# test_id: 714 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 960, 7, 7], dtype=torch.float16)])
# test_id: 715 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 992, 7, 7], dtype=torch.float16)])
# test_id: 716 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1024, 7, 7], dtype=torch.float16)])
# test_id: 717 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1056, 7, 7], dtype=torch.float16)])
# test_id: 718 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1088, 7, 7], dtype=torch.float16)])
# test_id: 719 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1120, 7, 7], dtype=torch.float16)])
# test_id: 720 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1152, 7, 7], dtype=torch.float16)])
# test_id: 721 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1184, 7, 7], dtype=torch.float16)])
# test_id: 722 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1216, 7, 7], dtype=torch.float16)])
# test_id: 723 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1248, 7, 7], dtype=torch.float16)])
# test_id: 724 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1280, 7, 7], dtype=torch.float16)])
# test_id: 725 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1312, 7, 7], dtype=torch.float16)])
# test_id: 726 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1344, 7, 7], dtype=torch.float16)])
# test_id: 727 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1376, 7, 7], dtype=torch.float16)])
# test_id: 728 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1408, 7, 7], dtype=torch.float16)])
# test_id: 729 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1440, 7, 7], dtype=torch.float16)])
# test_id: 730 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1472, 7, 7], dtype=torch.float16)])
# test_id: 731 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1504, 7, 7], dtype=torch.float16)])
# test_id: 732 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1536, 7, 7], dtype=torch.float16)])
# test_id: 733 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1568, 7, 7], dtype=torch.float16)])
# test_id: 734 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1600, 7, 7], dtype=torch.float16)])
# test_id: 735 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1632, 7, 7], dtype=torch.float16)])
# test_id: 736 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1664, 7, 7], dtype=torch.float16)])
# test_id: 737 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1696, 7, 7], dtype=torch.float16)])
# test_id: 738 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1728, 7, 7], dtype=torch.float16)])
# test_id: 739 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1760, 7, 7], dtype=torch.float16)])
# test_id: 740 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1792, 7, 7], dtype=torch.float16)])
# test_id: 741 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1824, 7, 7], dtype=torch.float16)])
# test_id: 742 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1856, 7, 7], dtype=torch.float16)])
# test_id: 743 
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([1, 1888, 7, 7], dtype=torch.float16)])
