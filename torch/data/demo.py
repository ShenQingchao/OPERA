# test_id: 250
para_0 = torch.randn([1, 4], dtype=torch.float64)
para_1 = torch.randn([128, 4], dtype=torch.float64)
para_2 = torch.randn([128], dtype=torch.float64)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)