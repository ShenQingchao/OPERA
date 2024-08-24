import copy


def get_abstract_int(int_v):
    # convert a int value into an abstract int
    # converted rule:
    # [None, < -1, -1, 0, 1, > 1]  encode --> [-100, -2, -1, 0, 1, 2]
    if int_v is None:
        new_int = -100
    elif int_v < -1:
        new_int = -2
    elif int_v in [-1, 0, 1]:
        new_int = int_v
    elif int_v > 1:
        new_int = 2
    else:
        raise Exception(f"The value '{int_v}' cannot be abstracted")
    return new_int


def get_abstract_list(_list):
    # convert a list into an abstract list.
    # converted rule:
    # [None, < -1, -1, 0, 1, > 1]  encode --> [-100, -2, -1, 0, 1, 2]

    # add some additional element to express the property of the list
    # 0 for equal, 1 for greater, -1 for less or not equal (for list/tuple)
    abstract_list = []
    for i, e in enumerate(_list):
        if e is None:
            if i == 0:
                abstract_int = 2  # consider the first None in list is batch_size and set it as 2
            else:
                abstract_int = -100
            abstract_list.append(abstract_int)
        elif isinstance(e, int):
            abstract_int = get_abstract_int(e)
            abstract_list.append(abstract_int)
        elif isinstance(e, list) or isinstance(e, tuple):
            abstract_list.append(get_abstract_list(e))
        elif isinstance(e, str) or isinstance(e, bool):
            abstract_list.append(e)
        elif isinstance(e, float):
            new_e = 0.1 if e > 0 else -0.1
            abstract_list.append(new_e)
        elif isinstance(e, dict):
            abstract_list.append(str(e))
        else:
            raise TypeError(f"Unsupported type {type(e)} in {_list}!")

    if len(_list) == 2:
        e0 = _list[0]
        e1 = _list[1]
        if e0 == e1:
            abstract_list.append(0)
        elif isinstance(e0, int) and isinstance(e1, int):
            if e0 < e1:
                abstract_list.append(-1)
            elif e0 > e1:
                abstract_list.append(1)
            else:
                abstract_list.append(0)
        else:
            abstract_list.append(-1)

    return abstract_list


if __name__ == '__main__':
    # a = [9, -3]
    a = [[[1, 2], [3, 4]]]
    res = get_abstract_list(a)
    print(res)
