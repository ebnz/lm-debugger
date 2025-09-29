def flatten_list(input_list):
    return [item for inner_list in input_list for item in inner_list]


def round_struct_recursively(struct, decimals=3):
    if isinstance(struct, float):
        return round(struct, decimals)
    elif isinstance(struct, dict):
        rv = {}
        for key in struct.keys():
            rv[key] = round_struct_recursively(struct[key], decimals=decimals)
        return rv
    elif isinstance(struct, list):
        rv = []
        for idx in range(len(struct)):
            rv.append(round_struct_recursively(struct[idx], decimals=decimals))
        return rv
    else:
        return struct
