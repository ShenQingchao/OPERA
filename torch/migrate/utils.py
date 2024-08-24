class GlobalVar:
    count = 0
    all_api_call = []

    @staticmethod
    def add_count():
        GlobalVar.count += 1

    @staticmethod
    def add_call(api_call_str):
        GlobalVar.all_api_call.append(api_call_str)
