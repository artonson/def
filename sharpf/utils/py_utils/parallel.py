import os

from joblib import Parallel, delayed


def parallel_parameterised(func, iterable, backend='threading', n_threads=None):
    if None is n_threads:
        n_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    parallel = Parallel(n_jobs=n_threads, backend=backend)

    def ensure_tuple(args):
        return args if isinstance(args, tuple) else (args,)

    delayed_iterable = (delayed(func)(*ensure_tuple(args)) for args in iterable)
    return parallel(delayed_iterable)


def threaded_parallel(func, iterable):
    return parallel_parameterised(func, iterable, backend='threading')


def multiproc_parallel(func, iterable):
    return parallel_parameterised(func, iterable, backend='multiprocessing')
