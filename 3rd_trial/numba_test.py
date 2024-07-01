import math
import threading
from timeit import repeat

import numpy as np
from numba import jit

nthreads = 128
size = 10**10


def func_np(a, b):
    """
    Control function using Numpy.
    """
    return np.exp(2.1 * a + 3.2 * b)


@jit("void(double[:], double[:], double[:])", nopython=True, nogil=True)
def inner_func_nb(result, a, b):
    """
    Function under test.
    """
    for i in range(len(result)):
        result[i] = math.exp(2.1 * a[i] + 3.2 * b[i])


def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before the benchmark is
    # started
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print(
        "{:>5.0f} ms".format(
            min(repeat(lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000
        )
    )
    return res


def make_singlethread(inner_func):
    """
    Run the given function inside a single thread.
    """

    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result

    return func


def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """

    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [
            [arg[i * chunklen : (i + 1) * chunklen] for arg in args]
            for i in range(numthreads)
        ]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk) for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result

    return func_mt


func_nb = make_singlethread(inner_func_nb)
func_nb_mt = make_multithread(inner_func_nb, nthreads)

a = np.random.rand(size)
b = np.random.rand(size)

correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
timefunc(correct, "numba (1 thread)", func_nb, a, b)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, a, b)

import multiprocessing
import time
import logging

# Configure logging
logging.basicConfig(filename='cpu_benchmark.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def cpu_bound_task(n):
    """
    A CPU-bound task that sums numbers from 0 to n.
    """
    total = 0
    for i in range(n):
        total += i**i
    return total

def worker_process(n, result_queue):
    """
    Worker process that performs the CPU-bound task and puts the result in a queue.
    """
    result = cpu_bound_task(n)
    result_queue.put(result)

if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()
    n = 10**10  # Adjust this number based on how intensive you want the task to be

    processes = []
    result_queue = multiprocessing.Queue()

    start_time = time.time()
    logging.info(f"Starting CPU benchmark with {num_processes} processes")

    # Start the processes
    for _ in range(num_processes):
        process = multiprocessing.Process(target=worker_process, args=(n, result_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    end_time = time.time()

    # Collect results
    results = [result_queue.get() for _ in range(num_processes)]

    logging.info(f"Results: {results}")
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logging.info(f"CPU count: {num_processes}")
    logging.info(f"Max CPU utilization tested.")

    # Print to console as well
    print(f"Results: {results}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"CPU count: {num_processes}")
    print(f"Max CPU utilization tested.")

    # Keep the program running for a while to observe CPU usage
    time.sleep(10)
