import os

from joblib import Parallel, delayed


def omp_parallel(func, iterable, backend='threading'):
    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    parallel = Parallel(n_jobs=n_omp_threads, backend=backend)

    def ensure_tuple(args):
        return args if isinstance(args, tuple) else (args,)

    delayed_iterable = (delayed(func)(*ensure_tuple(args)) for args in iterable)
    return parallel(delayed_iterable)


def threaded_parallel(func, iterable):
    return omp_parallel(func, iterable, backend='threading')


def multiproc_parallel(func, iterable):
    return omp_parallel(func, iterable, backend='multiprocessing')
