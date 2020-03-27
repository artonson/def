import os

from joblib import Parallel, delayed


def threaded_parallel(func, iterable):
    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    parallel = Parallel(n_jobs=n_omp_threads, backend='threading')
    delayed_iterable = (delayed(func)(*args) for args in iterable)
    return parallel(delayed_iterable)
