# test_id: 0 
para_0 = torch.randn([1, 2, 8, 9, 10], dtype=torch.float32)
para_1 = [5, 7, 9]
class adaptive_avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool3d(args[0], para_1,)
verify_model(adaptive_avg_pool3d().float().eval(), input_data=para_0)


# test_id: 1 
para_0 = torch.randn([2, 8, 9, 10], dtype=torch.float32)
para_1 = [5, 7, 9]
class adaptive_avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool3d(args[0], para_1,)
verify_model(adaptive_avg_pool3d().float().eval(), input_data=para_0)


# test_id: 2 
para_0 = torch.randn([1, 2, 8, 9, 10], dtype=torch.float32)
para_1 = 7
class adaptive_avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool3d(args[0], para_1,)
verify_model(adaptive_avg_pool3d().float().eval(), input_data=para_0)


# test_id: 3 
para_0 = torch.randn([2, 8, 9, 10], dtype=torch.float32)
para_1 = 7
class adaptive_avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool3d(args[0], para_1,)
verify_model(adaptive_avg_pool3d().float().eval(), input_data=para_0)


# test_id: 4 
para_0 = torch.randn([2, 8, 9, 10], dtype=torch.float32)
para_1 = [7, 9]
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 5 
para_0 = torch.randn([8, 9, 10], dtype=torch.float32)
para_1 = [7, 9]
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 6 
para_0 = torch.randn([2, 8, 9, 10], dtype=torch.float32)
para_1 = 7
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 7 
para_0 = torch.randn([8, 9, 10], dtype=torch.float32)
para_1 = 7
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], para_1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)


# test_id: 8 
para_0 = torch.randn([8, 9, 10], dtype=torch.float32)
para_1 = 7
class adaptive_avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool1d(args[0], para_1,)
verify_model(adaptive_avg_pool1d().float().eval(), input_data=para_0)


# test_id: 9 
para_0 = torch.randn([9, 10], dtype=torch.float32)
para_1 = 7
class adaptive_avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool1d(args[0], para_1,)
verify_model(adaptive_avg_pool1d().float().eval(), input_data=para_0)


# test_id: 10 
para_0 = torch.randn([2, 1, 1, 4, 4], dtype=torch.float32)
para_1 = [2, 2, 2]
para_2 = False
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 11 
para_0 = torch.randn([4, 1, 3, 32, 32], dtype=torch.float32)
para_1 = [2, 2, 2]
para_2 = False
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 12 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [2, 2, 2]
para_2 = False
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 13 
para_0 = torch.randn([2, 1, 1, 4, 4], dtype=torch.float32)
para_1 = [4, 4, 4]
para_2 = False
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 14 
para_0 = torch.randn([4, 1, 3, 32, 32], dtype=torch.float32)
para_1 = [4, 4, 4]
para_2 = False
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 15 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [4, 4, 4]
para_2 = False
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 16 
para_0 = torch.randn([2, 1, 1, 4, 4], dtype=torch.float32)
para_1 = [2, 2, 2]
para_2 = True
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 17 
para_0 = torch.randn([4, 1, 3, 32, 32], dtype=torch.float32)
para_1 = [2, 2, 2]
para_2 = True
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 18 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [2, 2, 2]
para_2 = True
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 19 
para_0 = torch.randn([2, 1, 1, 4, 4], dtype=torch.float32)
para_1 = [4, 4, 4]
para_2 = True
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 20 
para_0 = torch.randn([4, 1, 3, 32, 32], dtype=torch.float32)
para_1 = [4, 4, 4]
para_2 = True
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 21 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [4, 4, 4]
para_2 = True
class adaptive_max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool3d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool3d().float().eval(), input_data=para_0)


# test_id: 22 
para_0 = torch.randn([2, 1, 4, 4], dtype=torch.float32)
para_1 = [2, 2]
para_2 = False
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 23 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [2, 2]
para_2 = False
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 24 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = [2, 2]
para_2 = False
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 25 
para_0 = torch.randn([2, 1, 4, 4], dtype=torch.float32)
para_1 = [4, 4]
para_2 = False
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 26 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [4, 4]
para_2 = False
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 27 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = [4, 4]
para_2 = False
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 28 
para_0 = torch.randn([2, 1, 4, 4], dtype=torch.float32)
para_1 = [2, 2]
para_2 = True
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 29 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [2, 2]
para_2 = True
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 30 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = [2, 2]
para_2 = True
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 31 
para_0 = torch.randn([2, 1, 4, 4], dtype=torch.float32)
para_1 = [4, 4]
para_2 = True
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 32 
para_0 = torch.randn([1, 3, 32, 32], dtype=torch.float32)
para_1 = [4, 4]
para_2 = True
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 33 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = [4, 4]
para_2 = True
class adaptive_max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool2d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool2d().float().eval(), input_data=para_0)


# test_id: 34 
para_0 = torch.randn([1, 4, 4], dtype=torch.float32)
para_1 = 2
para_2 = False
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 35 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = 2
para_2 = False
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 36 
para_0 = torch.randn([16, 8], dtype=torch.float32)
para_1 = 2
para_2 = False
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 37 
para_0 = torch.randn([1, 4, 4], dtype=torch.float32)
para_1 = 4
para_2 = False
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 38 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = 4
para_2 = False
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 39 
para_0 = torch.randn([16, 8], dtype=torch.float32)
para_1 = 4
para_2 = False
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 40 
para_0 = torch.randn([1, 4, 4], dtype=torch.float32)
para_1 = 2
para_2 = True
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 41 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = 2
para_2 = True
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 42 
para_0 = torch.randn([16, 8], dtype=torch.float32)
para_1 = 2
para_2 = True
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 43 
para_0 = torch.randn([1, 4, 4], dtype=torch.float32)
para_1 = 4
para_2 = True
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 44 
para_0 = torch.randn([3, 32, 32], dtype=torch.float32)
para_1 = 4
para_2 = True
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 45 
para_0 = torch.randn([16, 8], dtype=torch.float32)
para_1 = 4
para_2 = True
class adaptive_max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_max_pool1d(args[0], para_1,para_2,)
verify_model(adaptive_max_pool1d().float().eval(), input_data=para_0)


# test_id: 46 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 47 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 48 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 49 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 50 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 51 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 52 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 53 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 54 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 55 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 56 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 57 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 58 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 59 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 60 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 61 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 62 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 63 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 64 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 65 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 66 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 67 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 68 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 69 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 70 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 71 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 72 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 73 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 74 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 75 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 76 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 77 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 78 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 79 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 80 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 81 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 82 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 83 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 84 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 85 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 86 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 87 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 88 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 89 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 90 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 91 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 92 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 93 
para_0 = torch.randn([20, 6, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 94 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 95 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 96 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 97 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 98 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 99 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 100 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 101 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 102 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 103 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 104 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 105 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 106 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 107 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 108 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 109 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 110 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 111 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 112 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 113 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 114 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 115 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 116 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 117 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 118 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 119 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 120 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 121 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 122 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 123 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 124 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 125 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 126 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 127 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 128 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 129 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 130 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 131 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 132 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 133 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 134 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 135 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 136 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 137 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 138 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 139 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 140 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 141 
para_0 = torch.randn([20, 6, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 142 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 143 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 144 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 145 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 146 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 147 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 148 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 149 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 150 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 151 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 152 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 153 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 154 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 155 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 156 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 157 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = None
para_2 = None
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 158 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 159 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 160 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 161 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 162 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 163 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 164 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 165 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 166 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 167 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 168 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 169 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 170 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 171 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 172 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 173 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=True,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 174 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 175 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 176 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 177 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=1.0,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 178 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 179 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 180 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 181 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=5e-05,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 182 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 183 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 184 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 185 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.5,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 186 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 187 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = torch.randn([6], dtype=torch.float32)
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 188 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = torch.randn([6], dtype=torch.float32)
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 189 
para_0 = torch.randn([20, 6, 10, 10, 10], dtype=torch.float32)
para_1 = torch.randn([6], dtype=torch.float32)
para_2 = torch.randn([6], dtype=torch.float32)
para_3 = None
para_4 = None
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,eps=0.042,training=False,)
verify_model(batch_norm().float().eval(), input_data=para_0)


# test_id: 190 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 191 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 2
para_4 = 0
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 192 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 1
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 193 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 2
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 194 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [0, 1]
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 195 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [1, 0]
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 196 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 'same'
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 197 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 'valid'
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 198 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([3, 1, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 3
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 199 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 200 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 2
para_4 = 0
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 201 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 1
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 202 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 2
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 203 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [0, 1]
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 204 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [1, 0]
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 205 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 'same'
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 206 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 'valid'
para_5 = 1
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 207 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([3, 1, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 3
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 208 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 209 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 2
para_4 = 0
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 210 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 1
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 211 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 2
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 212 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 'same'
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 213 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 'valid'
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 214 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 1, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 3
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 215 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 216 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 2
para_4 = 0
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 217 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 1
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 218 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 2
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 219 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 'same'
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 220 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 'valid'
para_5 = 1
para_6 = 1
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 221 
para_0 = torch.randn([2, 3, 25], dtype=torch.float32)
para_1 = torch.randn([3, 1, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 3
class conv1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv1d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv1d().float().eval(), input_data=para_0)


# test_id: 222 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 223 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 2
para_4 = 0
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 224 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 1
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 225 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 2
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 226 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [0, 1, 0]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 227 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [1, 0, 0]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 228 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [0, 0, 1]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 229 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [1, 1, 0]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 230 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [0, 1, 1]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 231 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = [1, 0, 1]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 232 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 'same'
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 233 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([1], dtype=torch.float32)
para_3 = 1
para_4 = 'valid'
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 234 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([3, 1, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 3
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 235 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 236 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 2
para_4 = 0
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 237 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 1
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 238 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 2
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 239 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [0, 1, 0]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 240 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [1, 0, 0]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 241 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [0, 0, 1]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 242 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [1, 1, 0]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 243 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [0, 1, 1]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 244 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = [1, 0, 1]
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 245 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 'same'
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 246 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([1, 3, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 'valid'
para_5 = 1
para_6 = 1
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 247 
para_0 = torch.randn([2, 3, 25, 25, 25], dtype=torch.float32)
para_1 = torch.randn([3, 1, 3, 3, 3], dtype=torch.float32)
para_2 = None
para_3 = 1
para_4 = 0
para_5 = 1
para_6 = 3
class conv3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv3d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv3d().float().eval(), input_data=para_0)


# test_id: 248 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
para_1 = torch.randn([3, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)


# test_id: 249 
verify_model(torch.nn.Conv2d(3,3,3,1,1,dilation=1,groups=1,bias=True,).eval(), input_data=[torch.randn([2, 3, 25, 25], dtype=torch.float32)])
# test_id: 250 
para_0 = torch.randn([2, 3, 25, 25], dtype=torch.float32)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], inplace=True,)
verify_model(relu().float().eval(), input_data=para_0)


# test_id: 251 
verify_model(torch.nn.ReLU(inplace=True,).eval(), input_data=[torch.randn([2, 3, 25, 25], dtype=torch.float32)])
# test_id: 252 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1],padding=[0, 0],output_padding=[0, 0],dilation=[2, 2],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 253 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=[1, 1],padding=[0, 0],output_padding=[0, 0],dilation=[1, 1],groups=3,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 254 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1],padding=[1, 1],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 255 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1],padding=[3, 1],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 256 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1],padding=[1, 0],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 257 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=[1, 1],padding=[1, 0],output_padding=[0, 0],dilation=[1, 1],groups=3,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 258 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=[1, 1],padding=[1, 0],output_padding=[0, 0],dilation=[2, 2],groups=3,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 259 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[2, 1],padding=[1, 0],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 260 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[2, 2],padding=[0, 0],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 261 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[2, 2],padding=[1, 1],output_padding=[1, 1],dilation=[2, 2],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 262 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[0, 0],output_padding=[0, 0],dilation=[2, 2],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 263 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[0, 0],output_padding=[0, 0],dilation=[1, 1],groups=3,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 264 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[1, 1],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 265 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[3, 1],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 266 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[1, 0],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 267 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[1, 0],output_padding=[0, 0],dilation=[1, 1],groups=3,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 268 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1],padding=[1, 0],output_padding=[0, 0],dilation=[2, 2],groups=3,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 269 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[2, 1],padding=[1, 0],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 270 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[2, 2],padding=[0, 0],output_padding=[0, 0],dilation=[1, 1],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 271 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], weight={'shape': [3, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[2, 2],padding=[1, 1],output_padding=[1, 1],dilation=[2, 2],groups=1,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)


# test_id: 272 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=1,padding=0,output_padding=0,dilation=1,groups=1,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 273 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=1,padding=0,output_padding=0,dilation=1,groups=3,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 274 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=1,padding=1,output_padding=0,dilation=1,groups=1,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 275 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=1,padding=1,output_padding=0,dilation=1,groups=3,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 276 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=1,padding=3,output_padding=1,dilation=2,groups=1,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 277 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=1,padding=3,output_padding=1,dilation=2,groups=3,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 278 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=1,padding=0,output_padding=0,dilation=1,groups=1,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 279 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=1,padding=0,output_padding=0,dilation=1,groups=3,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 280 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=1,padding=1,output_padding=0,dilation=1,groups=1,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 281 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=1,padding=1,output_padding=0,dilation=1,groups=3,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 282 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=1,padding=3,output_padding=1,dilation=2,groups=1,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 283 
para_0 = torch.randn([1, 3, 10], dtype=torch.float32)
class conv_transpose1d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose1d(args[0], weight={'shape': [3, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=1,padding=3,output_padding=1,dilation=2,groups=3,)
verify_model(conv_transpose1d().float().eval(), input_data=para_0)


# test_id: 284 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1, 1],padding=[0, 0, 0],output_padding=[0, 0, 0],dilation=[2, 2, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 285 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=[1, 1, 1],padding=[0, 0, 0],output_padding=[0, 0, 0],dilation=[1, 1, 1],groups=3,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 286 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1, 1],padding=[1, 1, 0],output_padding=[0, 0, 1],dilation=[1, 1, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 287 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1, 2],padding=[3, 1, 0],output_padding=[1, 1, 1],dilation=[4, 4, 4],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 288 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[1, 1, 1],padding=[1, 0, 1],output_padding=[0, 1, 0],dilation=[1, 2, 1],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 289 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=[1, 1, 1],padding=[1, 0, 0],output_padding=[0, 0, 0],dilation=[1, 1, 2],groups=3,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 290 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [3], 'dtype': 'torch.float32'},stride=[1, 1, 1],padding=[1, 0, 0],output_padding=[0, 0, 0],dilation=[2, 2, 1],groups=3,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 291 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[2, 1, 2],padding=[1, 0, 0],output_padding=[2, 0, 0],dilation=[3, 4, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 292 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[2, 2, 2],padding=[0, 0, 0],output_padding=[0, 0, 0],dilation=[1, 1, 1],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 293 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias={'shape': [1], 'dtype': 'torch.float32'},stride=[2, 2, 2],padding=[1, 1, 2],output_padding=[1, 1, 0],dilation=[2, 2, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 294 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 1],padding=[0, 0, 0],output_padding=[0, 0, 0],dilation=[2, 2, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 295 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 1],padding=[0, 0, 0],output_padding=[0, 0, 0],dilation=[1, 1, 1],groups=3,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 296 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 1],padding=[1, 1, 0],output_padding=[0, 0, 1],dilation=[1, 1, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 297 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 2],padding=[3, 1, 0],output_padding=[1, 1, 1],dilation=[4, 4, 4],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 298 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 1],padding=[1, 0, 1],output_padding=[0, 1, 0],dilation=[1, 2, 1],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 299 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 1],padding=[1, 0, 0],output_padding=[0, 0, 0],dilation=[1, 1, 2],groups=3,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 300 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[1, 1, 1],padding=[1, 0, 0],output_padding=[0, 0, 0],dilation=[2, 2, 1],groups=3,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 301 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[2, 1, 2],padding=[1, 0, 0],output_padding=[2, 0, 0],dilation=[3, 4, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 302 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[2, 2, 2],padding=[0, 0, 0],output_padding=[0, 0, 0],dilation=[1, 1, 1],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 303 
para_0 = torch.randn([1, 3, 10, 10, 4], dtype=torch.float32)
class conv_transpose3d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose3d(args[0], weight={'shape': [3, 1, 1, 1, 1], 'dtype': 'torch.float32'},bias=None,stride=[2, 2, 2],padding=[1, 1, 2],output_padding=[1, 1, 0],dilation=[2, 2, 2],groups=1,)
verify_model(conv_transpose3d().float().eval(), input_data=para_0)


# test_id: 304 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 2.0
para_3 = 1e-06
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 305 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 4.0
para_3 = 1e-06
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 306 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 6.0
para_3 = 1e-06
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 307 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 8.0
para_3 = 1e-06
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 308 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 2.0
para_3 = 1e-05
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 309 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 4.0
para_3 = 1e-05
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 310 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 6.0
para_3 = 1e-05
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 311 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 8.0
para_3 = 1e-05
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 312 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 2.0
para_3 = 1e-07
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 313 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 4.0
para_3 = 1e-07
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 314 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 6.0
para_3 = 1e-07
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 315 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 8.0
para_3 = 1e-07
para_4 = True
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 316 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 2.0
para_3 = 1e-06
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 317 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 4.0
para_3 = 1e-06
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 318 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 6.0
para_3 = 1e-06
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 319 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 8.0
para_3 = 1e-06
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 320 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 2.0
para_3 = 1e-05
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 321 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 4.0
para_3 = 1e-05
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 322 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 6.0
para_3 = 1e-05
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 323 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 8.0
para_3 = 1e-05
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 324 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 2.0
para_3 = 1e-07
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 325 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 4.0
para_3 = 1e-07
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 326 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 6.0
para_3 = 1e-07
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 327 
para_0 = torch.randn([20, 100], dtype=torch.float32)
para_1 = torch.randn([20, 100], dtype=torch.float32)
para_2 = 8.0
para_3 = 1e-07
para_4 = False
class pairwise_distance(Module):
    def forward(self, *args):
        return torch.nn.functional.pairwise_distance(args[0], para_1,para_2,para_3,para_4,)
verify_model(pairwise_distance().float().eval(), input_data=para_0)


# test_id: 328 
para_0 = torch.randint(1, 100, [8], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
para_2 = torch.randint(1, 100, [2], dtype=torch.int64)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,para_2,mode='sum',per_sample_weights={'shape': [8], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 329 
para_0 = torch.randint(1, 100, [8], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
para_2 = torch.randint(1, 100, [2], dtype=torch.int32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,para_2,mode='sum',per_sample_weights={'shape': [8], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 330 
para_0 = torch.randint(1, 100, [8], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
para_2 = torch.randint(1, 100, [2], dtype=torch.int64)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,para_2,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 331 
para_0 = torch.randint(1, 100, [8], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
para_2 = torch.randint(1, 100, [2], dtype=torch.int32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,para_2,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 332 
para_0 = torch.randint(1, 100, [1, 1], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [1, 1], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 333 
para_0 = torch.randint(1, 100, [2, 5], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [2, 5], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 334 
para_0 = torch.randint(1, 100, [3, 10], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [3, 10], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 335 
para_0 = torch.randint(1, 100, [4, 7], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [4, 7], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 336 
para_0 = torch.randint(1, 100, [1, 1], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [1, 1], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 337 
para_0 = torch.randint(1, 100, [2, 5], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [2, 5], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 338 
para_0 = torch.randint(1, 100, [3, 10], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [3, 10], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 339 
para_0 = torch.randint(1, 100, [4, 7], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',per_sample_weights={'shape': [4, 7], 'dtype': 'torch.float32'},)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 340 
para_0 = torch.randint(1, 100, [1, 1], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 341 
para_0 = torch.randint(1, 100, [2, 5], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 342 
para_0 = torch.randint(1, 100, [3, 10], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 343 
para_0 = torch.randint(1, 100, [4, 7], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 344 
para_0 = torch.randint(1, 100, [1, 1], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 345 
para_0 = torch.randint(1, 100, [2, 5], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 346 
para_0 = torch.randint(1, 100, [3, 10], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 347 
para_0 = torch.randint(1, 100, [4, 7], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,mode='sum',)
verify_model(embedding_bag().float().eval(), input_data=para_0)


# test_id: 348 
para_0 = torch.randint(1, 100, [1], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 349 
para_0 = torch.randint(1, 100, [2], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 350 
para_0 = torch.randint(1, 100, [3], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 351 
para_0 = torch.randint(1, 100, [4], dtype=torch.int64)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 352 
para_0 = torch.randint(1, 100, [1], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 353 
para_0 = torch.randint(1, 100, [2], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 354 
para_0 = torch.randint(1, 100, [3], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 355 
para_0 = torch.randint(1, 100, [4], dtype=torch.int32)
para_1 = torch.randn([10, 10], dtype=torch.float32)
class embedding(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding(args[0], para_1,)
verify_model(embedding().float().eval(), input_data=para_0)


# test_id: 356 
para_0 = torch.randn([2, 3], dtype=torch.float32)
class gelu(Module):
    def forward(self, *args):
        return torch.nn.functional.gelu(args[0], approximate='none',)
verify_model(gelu().float().eval(), input_data=para_0)


# test_id: 357 
para_0 = torch.randn([2, 3], dtype=torch.float32)
class gelu(Module):
    def forward(self, *args):
        return torch.nn.functional.gelu(args[0], approximate='tanh',)
verify_model(gelu().float().eval(), input_data=para_0)


# test_id: 358 
para_0 = torch.randn([2, 4, 224, 224], dtype=torch.float32)
para_1 = 0
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], para_1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 359 
para_0 = torch.randn([2, 4, 224, 224], dtype=torch.float32)
para_1 = 1
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], para_1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 360 
para_0 = torch.randn([2, 4, 224, 224], dtype=torch.float32)
para_1 = 2
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], para_1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 361 
para_0 = torch.randn([2, 4, 224, 224], dtype=torch.float32)
para_1 = 3
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], para_1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 362 
para_0 = torch.randn([2, 4, 224, 224], dtype=torch.float32)
para_1 = -1
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], para_1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 363 
para_0 = torch.randn([2, 4, 224, 224], dtype=torch.float32)
para_1 = -2
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], para_1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 364 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'zeros'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 365 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'zeros'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 366 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'zeros'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 367 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'zeros'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 368 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'zeros'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 369 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'zeros'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 370 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'border'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 371 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'border'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 372 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'border'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 373 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'border'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 374 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'border'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 375 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'border'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 376 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'reflection'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 377 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'reflection'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 378 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'reflection'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 379 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'reflection'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 380 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'reflection'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 381 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'reflection'
para_4 = True
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 382 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'zeros'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 383 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'zeros'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 384 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'zeros'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 385 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'zeros'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 386 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'zeros'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 387 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'zeros'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 388 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'border'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 389 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'border'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 390 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'border'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 391 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'border'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 392 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'border'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 393 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'border'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 394 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'reflection'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 395 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'reflection'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 396 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'reflection'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 397 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'reflection'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 398 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'reflection'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 399 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'reflection'
para_4 = False
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 400 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'zeros'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 401 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'zeros'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 402 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'zeros'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 403 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'zeros'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 404 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'zeros'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 405 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'zeros'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 406 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'border'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 407 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'border'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 408 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'border'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 409 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'border'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 410 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'border'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 411 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'border'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 412 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'reflection'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 413 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bilinear'
para_3 = 'reflection'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 414 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'reflection'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 415 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'nearest'
para_3 = 'reflection'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 416 
para_0 = torch.randn([1, 3, 10, 10], dtype=torch.float32)
para_1 = torch.randn([1, 5, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'reflection'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 417 
para_0 = torch.randn([1, 3, 10, 15], dtype=torch.float32)
para_1 = torch.randn([1, 3, 5, 2], dtype=torch.float32)
para_2 = 'bicubic'
para_3 = 'reflection'
para_4 = None
class grid_sample(Module):
    def forward(self, *args):
        return torch.nn.functional.grid_sample(args[0], para_1,para_2,para_3,para_4,)
verify_model(grid_sample().float().eval(), input_data=para_0)


# test_id: 418 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 419 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 420 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 421 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 422 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 423 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 424 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 425 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 426 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 427 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 428 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 429 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 430 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 431 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 432 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 433 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 434 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 435 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 436 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 437 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=0,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 438 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 439 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 440 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 441 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 442 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 443 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 444 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 445 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 446 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 447 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 448 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 449 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 450 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 451 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 452 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 453 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 454 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 455 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 456 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 457 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=5,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 458 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 459 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 460 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 461 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 462 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 463 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 464 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 465 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 466 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 467 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 468 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 469 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 470 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 471 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 472 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 473 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 474 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 475 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 476 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 477 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=1,stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 478 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 479 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 480 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 481 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 482 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 483 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 484 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 485 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 486 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 487 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 488 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 489 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 490 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 491 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 492 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 493 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 494 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 495 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 496 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 497 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=[2, 3],stride=3,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 498 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 499 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 500 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 501 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 502 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 503 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 504 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 505 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 506 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 507 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 508 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 509 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 510 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 511 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 512 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 513 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 514 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 515 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 516 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 517 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=0,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 518 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 519 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 520 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 521 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 522 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 523 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 524 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 525 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 526 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 527 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 528 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 529 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 530 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 531 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 532 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 533 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 534 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 535 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 536 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 537 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=5,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 538 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 539 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 540 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 541 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 542 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 543 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 544 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 545 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 546 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 547 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 548 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 549 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 550 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 551 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 552 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 553 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 554 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 555 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 556 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 557 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=1,stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 558 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 559 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 560 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 561 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 562 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 563 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 564 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 565 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 566 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 567 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 568 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 569 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 570 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 571 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 572 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 573 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 574 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 575 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 576 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 577 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=[2, 3],stride=1,)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 578 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 579 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 580 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 581 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 582 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 583 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 584 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 585 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 586 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 587 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 588 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 589 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 590 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 591 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 592 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 593 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 594 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 595 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 596 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 597 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=0,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 598 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 599 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 600 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 601 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 602 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 603 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 604 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 605 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 606 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 607 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 608 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 609 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 610 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 611 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 612 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 613 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 614 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 615 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 616 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 617 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=5,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 618 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 619 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 620 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 621 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 622 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 623 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 624 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 625 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 626 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 627 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 628 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 629 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 630 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 631 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 632 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 633 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 634 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 635 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 636 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 637 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=1,stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 638 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=1,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 639 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=1,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 640 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=1,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 641 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=1,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 642 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=1,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 643 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=2,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 644 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=2,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 645 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=2,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 646 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=2,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 647 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=2,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 648 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=3,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 649 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=3,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 650 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=3,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 651 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=3,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 652 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=3,padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 653 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 3],dilation=(1, 2),padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 654 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 2],dilation=(1, 2),padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 655 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[3, 3],dilation=(1, 2),padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 656 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[2, 2],dilation=(1, 2),padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 657 
para_0 = torch.randn([10, 3, 24, 24], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=[1, 1],dilation=(1, 2),padding=[2, 3],stride=[2, 1],)
verify_model(unfold().float().eval(), input_data=para_0)


# test_id: 658 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = 0.01
class leaky_relu(Module):
    def forward(self, *args):
        return torch.nn.functional.leaky_relu(args[0], para_1,inplace=True,)
verify_model(leaky_relu().float().eval(), input_data=para_0)


# test_id: 659 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = 1.01
class leaky_relu(Module):
    def forward(self, *args):
        return torch.nn.functional.leaky_relu(args[0], para_1,inplace=True,)
verify_model(leaky_relu().float().eval(), input_data=para_0)


# test_id: 660 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = -0.01
class leaky_relu(Module):
    def forward(self, *args):
        return torch.nn.functional.leaky_relu(args[0], para_1,inplace=True,)
verify_model(leaky_relu().float().eval(), input_data=para_0)


# test_id: 661 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = 0.01
class leaky_relu(Module):
    def forward(self, *args):
        return torch.nn.functional.leaky_relu(args[0], para_1,inplace=False,)
verify_model(leaky_relu().float().eval(), input_data=para_0)


# test_id: 662 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = 1.01
class leaky_relu(Module):
    def forward(self, *args):
        return torch.nn.functional.leaky_relu(args[0], para_1,inplace=False,)
verify_model(leaky_relu().float().eval(), input_data=para_0)


# test_id: 663 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = -0.01
class leaky_relu(Module):
    def forward(self, *args):
        return torch.nn.functional.leaky_relu(args[0], para_1,inplace=False,)
verify_model(leaky_relu().float().eval(), input_data=para_0)


# test_id: 664 
para_0 = torch.randn([9], dtype=torch.float32)
para_1 = torch.randn([10, 9], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 665 
para_0 = torch.randn([9], dtype=torch.float32)
para_1 = torch.randn([9], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 666 
para_0 = torch.randn([3, 9], dtype=torch.float32)
para_1 = torch.randn([10, 9], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 667 
para_0 = torch.randn([3, 9], dtype=torch.float32)
para_1 = torch.randn([9], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 668 
para_0 = torch.randn([2, 3, 9], dtype=torch.float32)
para_1 = torch.randn([10, 9], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 669 
para_0 = torch.randn([2, 3, 9], dtype=torch.float32)
para_1 = torch.randn([9], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 670 
para_0 = torch.randn([9], dtype=torch.float32)
para_1 = torch.randn([10, 9], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 671 
para_0 = torch.randn([3, 9], dtype=torch.float32)
para_1 = torch.randn([10, 9], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 672 
para_0 = torch.randn([2, 3, 9], dtype=torch.float32)
para_1 = torch.randn([10, 9], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 673 
para_0 = torch.randn([1, 15, 10], dtype=torch.float32)
para_1 = torch.randn([66, 10], dtype=torch.float32)
para_2 = torch.randn([66], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)


# test_id: 674 
para_0 = torch.randint(1, 100, [5, 9, 7], dtype=torch.int64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=0,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 675 
para_0 = torch.randn([5, 9, 7], dtype=torch.float64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=0,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 676 
para_0 = torch.randn([5, 9, 7], dtype=torch.float64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=0,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 677 
para_0 = torch.randint(1, 100, [5, 9, 7], dtype=torch.int64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=1,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 678 
para_0 = torch.randn([5, 9, 7], dtype=torch.float64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=1,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 679 
para_0 = torch.randn([5, 9, 7], dtype=torch.float64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=1,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 680 
para_0 = torch.randint(1, 100, [5, 9, 7], dtype=torch.int64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=-1,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 681 
para_0 = torch.randn([5, 9, 7], dtype=torch.float64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=-1,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 682 
para_0 = torch.randn([5, 9, 7], dtype=torch.float64)
class log_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.log_softmax(args[0], dim=-1,dtype=None,)
verify_model(log_softmax().float().eval(), input_data=para_0)


# test_id: 683 
para_0 = torch.randint(1, 100, [1000], dtype=torch.int64)
para_1 = -1
class one_hot(Module):
    def forward(self, *args):
        return torch.nn.functional.one_hot(args[0], para_1,)
verify_model(one_hot().float().eval(), input_data=para_0)


# test_id: 684 
para_0 = torch.randint(1, 100, [1000], dtype=torch.int64)
para_1 = 3
class one_hot(Module):
    def forward(self, *args):
        return torch.nn.functional.one_hot(args[0], para_1,)
verify_model(one_hot().float().eval(), input_data=para_0)


# test_id: 685 
para_0 = torch.randint(1, 100, [1000], dtype=torch.int64)
para_1 = 1000
class one_hot(Module):
    def forward(self, *args):
        return torch.nn.functional.one_hot(args[0], para_1,)
verify_model(one_hot().float().eval(), input_data=para_0)


# test_id: 686 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 687 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = (1, 0, 0, 0, 0, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 688 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 689 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 690 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = (1, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 691 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (1, 0, 0, 0, 0, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 692 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 693 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 694 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int64)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 695 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 696 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = (1, 2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 697 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int8)
para_1 = (1, 0, 0, 0, 0, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 698 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int64)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 699 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int32)
para_1 = (1, 0, 0, 0, 0, 1, 1, 2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=0.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 700 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int32)
para_1 = (1, 2, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 701 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int64)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 702 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (0, 1, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 703 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = (0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 704 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int8)
para_1 = (0, 0, -1, -2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 705 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (-1, -2, -1, -2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 706 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = (-5, -8, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 707 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 2, 3, 4, 5, 6)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 708 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (1, 0, 0, 0, 0, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 709 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 710 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 711 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 2, 3, 4, 5, 6)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 712 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (1, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 713 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 0, 0, 0, 0, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 714 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 715 
para_0 = torch.randint(1, 100, [1, 3, 14, 14, 18], dtype=torch.int32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 716 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 717 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 718 
para_0 = torch.randint(1, 100, [1, 3, 14, 14, 18], dtype=torch.int64)
para_1 = (1, 2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 719 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 0, 0, 0, 0, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 720 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 721 
para_0 = torch.randint(1, 100, [1, 3, 14, 14, 18], dtype=torch.int8)
para_1 = (1, 0, 0, 0, 0, 1, 1, 2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=0.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 722 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 0, 0, 0, 0, 1, 1, 2, 2, 3)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=0.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 723 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (1, 2, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 724 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float64)
para_1 = (1, 2, 3, 4, 5, 6)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 725 
para_0 = torch.randint(1, 100, [1, 3, 14, 14, 18], dtype=torch.int32)
para_1 = (0, 1, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 726 
para_0 = torch.randint(1, 100, [1, 3, 14, 14, 18], dtype=torch.int64)
para_1 = (0, 0, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 727 
para_0 = torch.randint(1, 100, [1, 3, 14, 14, 18], dtype=torch.int8)
para_1 = (0, 0, -1, -2, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 728 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (-1, -2, -1, -2, -1, -2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 729 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (-5, -8, 0, 0, 0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 730 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = (10, 10, 10, 10, 10, 10)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='circular',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 731 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = (1, 2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 732 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = (1, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 733 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = (0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='reflect',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 734 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = (1, 2)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 735 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = (1, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 736 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = (0, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 737 
para_0 = torch.randint(1, 100, [1, 3], dtype=torch.int32)
para_1 = (1, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 738 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = (1, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 739 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = (1, 0)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 740 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 741 
para_0 = torch.randint(1, 100, [1, 3], dtype=torch.int64)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 742 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = (1, 2, 3, 4)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='constant',value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 743 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 744 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 745 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 746 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int64)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 747 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 748 
para_0 = torch.randint(1, 100, [1, 3, 14, 14], dtype=torch.int8)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 749 
para_0 = torch.randn([1, 3, 14, 14], dtype=torch.float64)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 750 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 751 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 752 
para_0 = torch.randn([1, 3, 14, 14, 18], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 753 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 754 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 755 
para_0 = torch.randint(1, 100, [1, 3], dtype=torch.int64)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 756 
para_0 = torch.randint(1, 100, [1, 3], dtype=torch.int32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=42.0,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 757 
para_0 = torch.randn([1, 3], dtype=torch.float32)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 758 
para_0 = torch.randint(1, 100, [1, 3], dtype=torch.int8)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=None,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 759 
para_0 = torch.randn([1, 3], dtype=torch.float64)
para_1 = [{'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}, {'shape': [], 'dtype': 'torch.int32'}]
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,value=-0.57,)
verify_model(pad().float().eval(), input_data=para_0)


# test_id: 760 
para_0 = torch.randn([1, 9, 4, 4], dtype=torch.float32)
para_1 = 3
class pixel_shuffle(Module):
    def forward(self, *args):
        return torch.nn.functional.pixel_shuffle(args[0], para_1,)
verify_model(pixel_shuffle().float().eval(), input_data=para_0)


# test_id: 761 
para_0 = torch.randn([1, 2, 3, 8, 4, 4], dtype=torch.float32)
para_1 = 2
class pixel_shuffle(Module):
    def forward(self, *args):
        return torch.nn.functional.pixel_shuffle(args[0], para_1,)
verify_model(pixel_shuffle().float().eval(), input_data=para_0)


# test_id: 762 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = True
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 763 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = True
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 764 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = True
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 765 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = True
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 766 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = False
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 767 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = False
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 768 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = False
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 769 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = False
para_5 = True
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 770 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = True
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 771 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = True
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 772 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = True
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 773 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = True
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 774 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = False
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 775 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = False
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 776 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = False
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 777 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = False
para_5 = False
class avg_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool1d().float().eval(), input_data=para_0)


# test_id: 778 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 779 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 780 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 781 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 782 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 783 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 784 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 785 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 786 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = True
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 787 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 788 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 789 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 790 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 791 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 792 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 793 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 794 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 795 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = False
para_5 = True
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 796 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 797 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 798 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 799 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 800 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 801 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 802 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 803 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 804 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = True
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 805 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 806 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 807 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 808 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 809 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 810 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 811 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 812 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 813 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = False
para_5 = False
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)


# test_id: 814 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = True
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 815 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = True
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 816 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = True
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 817 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = True
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 818 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = True
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 819 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = False
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 820 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = False
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 821 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = False
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 822 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = False
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 823 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = False
para_5 = True
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 824 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = True
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 825 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = True
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 826 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = True
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 827 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = True
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 828 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = True
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 829 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = False
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 830 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = False
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 831 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = False
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 832 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = False
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 833 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = False
para_5 = False
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)


# test_id: 834 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 835 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = 1
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 836 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = 1
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 837 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 1
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 838 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 839 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 840 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 841 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 842 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 843 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = 2
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 844 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = 2
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 845 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 2
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 846 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 847 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 848 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 849 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 850 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 851 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 852 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 853 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 854 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 855 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 856 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 857 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 858 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 859 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 860 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 861 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 862 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 863 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 864 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 865 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 866 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 867 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 868 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 869 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 870 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 871 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 872 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 873 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 874 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 875 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 876 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 877 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 878 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 879 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 880 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 881 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 882 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 883 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 884 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 885 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 886 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 887 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 888 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 889 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 890 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 891 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 892 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 893 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 894 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 895 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 896 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 897 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 898 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 899 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 900 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 901 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 902 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 903 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 904 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 905 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 906 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 907 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 1
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 908 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 909 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 910 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 911 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 1
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 912 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 913 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 2
para_5 = True
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 914 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 3
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 915 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = (4,)
para_2 = 1
para_3 = 1
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 916 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = (5,)
para_3 = 2
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 917 
para_0 = torch.randn([1, 3, 15], dtype=torch.float32)
para_1 = 4
para_2 = None
para_3 = 0
para_4 = 2
para_5 = False
class max_pool1d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool1d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool1d().float().eval(), input_data=para_0)


# test_id: 918 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 919 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 920 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 921 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 922 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 1
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 923 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 924 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 925 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 926 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 927 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 928 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 929 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 930 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 931 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = 1
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 932 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 933 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 934 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 935 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 936 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 2
para_5 = True
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 937 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 938 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = 1
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 939 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [0, 1]
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 940 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [1, 1]
para_3 = [1, 0]
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 941 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [3, 3]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 942 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = [2, 1]
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 943 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = None
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 944 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [2, 1]
para_2 = []
para_3 = 0
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 945 
para_0 = torch.randn([1, 3, 15, 15], dtype=torch.float32)
para_1 = [8, 8]
para_2 = [8, 4]
para_3 = 1
para_4 = 2
para_5 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)


# test_id: 946 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 947 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 948 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 949 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 1
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 950 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 951 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 952 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 953 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 954 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 1
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 955 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 956 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 957 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 958 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 2
para_5 = True
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 959 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = 1
para_3 = 0
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 960 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [1, 1, 1]
para_3 = 1
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 961 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 3, 3]
para_2 = [3, 3, 3]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 962 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = [3, 1, 1]
para_3 = [0, 0, 0]
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 963 
para_0 = torch.randn([1, 3, 15, 15, 15], dtype=torch.float32)
para_1 = [3, 2, 1]
para_2 = None
para_3 = [0, 0, 0]
para_4 = 2
para_5 = False
class max_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool3d(args[0], para_1,para_2,para_3,para_4,para_5,return_indices=True,)
verify_model(max_pool3d().float().eval(), input_data=para_0)


# test_id: 964 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = torch.randn([1], dtype=torch.float32)
class prelu(Module):
    def forward(self, *args):
        return torch.nn.functional.prelu(args[0], para_1,)
verify_model(prelu().float().eval(), input_data=para_0)


# test_id: 965 
para_0 = torch.randn([2, 3, 9], dtype=torch.quint8)
para_1 = -1.0
para_2 = 1.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 966 
verify_model(torch.nn.Hardtanh(inplace=True,).eval(), input_data=[torch.randn([2, 3, 9], dtype=torch.quint8)])
# test_id: 967 
para_0 = torch.randn([2, 3, 9], dtype=torch.quint8)
para_1 = -1.0
para_2 = 1.0
para_3 = False
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)


# test_id: 968 
verify_model(torch.nn.Hardtanh(inplace=False,).eval(), input_data=[torch.randn([2, 3, 9], dtype=torch.quint8)])
# test_id: 969 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
class selu(Module):
    def forward(self, *args):
        return torch.nn.functional.selu(args[0], inplace=True,)
verify_model(selu().float().eval(), input_data=para_0)


# test_id: 970 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
class selu(Module):
    def forward(self, *args):
        return torch.nn.functional.selu(args[0], inplace=False,)
verify_model(selu().float().eval(), input_data=para_0)


# test_id: 971 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
class silu(Module):
    def forward(self, *args):
        return torch.nn.functional.silu(args[0], )
verify_model(silu().float().eval(), input_data=para_0)


# test_id: 972 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = -1
class softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.softmax(args[0], para_1,dtype=None,)
verify_model(softmax().float().eval(), input_data=para_0)


# test_id: 973 
para_0 = torch.randn([1, 3, 224, 224], dtype=torch.float32)
para_1 = 3
class softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.softmax(args[0], para_1,dtype=None,)
verify_model(softmax().float().eval(), input_data=para_0)


# test_id: 974 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float32)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=0,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 975 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float32)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 976 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float32)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=2,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 977 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float32)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=3,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 978 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float32)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=-1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 979 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float32)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=-2,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 980 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float64)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=0,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 981 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float64)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 982 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float64)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=2,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 983 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float64)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=3,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 984 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float64)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=-1,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 985 
para_0 = torch.randn([2, 4, 10, 10], dtype=torch.float64)
class glu(Module):
    def forward(self, *args):
        return torch.nn.functional.glu(args[0], dim=-2,)
verify_model(glu().float().eval(), input_data=para_0)


# test_id: 986 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 987 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = 200
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 988 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 989 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 990 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='linear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 991 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = 200
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='linear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 992 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='linear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 993 
para_0 = torch.randn([1, 3, 224], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='linear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 994 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 995 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 996 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = (300, 400)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 997 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 998 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 999 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.5, 2),mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1000 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1001 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1002 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = (400, 480)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1003 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1004 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1005 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.2, 1.3),mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1006 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1007 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1008 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = (400, 480)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1009 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1010 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1011 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.2, 1.3),mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1012 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bilinear',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1013 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bilinear',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1014 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = (400, 480)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bilinear',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1015 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='bilinear',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1016 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='bilinear',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1017 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.2, 1.3),mode='bilinear',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1018 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 300
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bicubic',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1019 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bicubic',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1020 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = (400, 480)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='bicubic',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1021 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='bicubic',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1022 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='bicubic',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1023 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.2, 1.3),mode='bicubic',antialias=True,)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1024 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = 200
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1025 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1026 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = (150, 200, 250)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1027 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1028 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1029 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.5, 2, 2.5),mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1030 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = 200
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='trilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1031 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = 150
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='trilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1032 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = (200, 240, 210)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=None,mode='trilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1033 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=2.5,mode='trilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1034 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=0.75,mode='trilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1035 
para_0 = torch.randn([1, 3, 100, 100, 100], dtype=torch.float32)
para_1 = None
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], para_1,scale_factor=(1.2, 1.1, 1.5),mode='trilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1036 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], size=torch.Size([200, 200]),mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1037 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], size=torch.Size([200, 200]),mode='bilinear',)
verify_model(interpolate().float().eval(), input_data=para_0)


# test_id: 1038 
para_0 = torch.randn([1, 3, 200, 200], dtype=torch.float32)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], size=torch.Size([200, 200]),mode='bicubic',)
verify_model(interpolate().float().eval(), input_data=para_0)


