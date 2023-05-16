import multiprocessing
import psutil


def compute_optimal_cpu_count():
    result = max(multiprocessing.cpu_count() - 2, 1)
    return result
