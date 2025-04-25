from functools import wraps
import time
from random import randint
import numpy as np

def time_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - st
    return wrapper

@time_counter
def count_usual(n):
    arr = [randint(1, 100_000) for _ in range(n)]
    s = sum(arr)
    m = sum(arr) / len(arr)
    return s, m

@time_counter
def count_numpy(n):
    arr = np.random.randint(low=1, high=100_000, size=n)
    s = arr.sum
    m = arr.mean
    return s, m

N = 10**int(input("n=10^x - введите x\n"))
print(f"обычный python при n={N} => {count_usual(N)} s")
print(f"python с numpy при n={N} => {count_numpy(N)} s")
