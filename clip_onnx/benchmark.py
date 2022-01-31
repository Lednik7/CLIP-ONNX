import time
import torch


def speed_test(func, data_gen, n: int = 5, empty_cache: bool = True):
    if empty_cache:
        torch.cuda.empty_cache()
    values = []
    for _ in range(n):
        input_data = data_gen()
        t = time.time()
        func(input_data)
        values.append(time.time() - t)
        if empty_cache:
            torch.cuda.empty_cache()
    return sum(values) / n
