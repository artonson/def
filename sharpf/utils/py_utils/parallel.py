import os

from joblib import Parallel, delayed


def omp_parallel(func, iterable, backend='threading', **parallel_kwargs):
    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    parallel = Parallel(
        n_jobs=n_omp_threads,
        backend=backend,
        **parallel_kwargs)

    def ensure_tuple(args):
        return args if isinstance(args, tuple) else (args,)

    delayed_iterable = (delayed(func)(*ensure_tuple(args)) for args in iterable)
    return parallel(delayed_iterable)


def threaded_parallel(func, iterable, **parallel_kwargs):
    return omp_parallel(func, iterable, backend='threading', **parallel_kwargs)


def multiproc_parallel(func, iterable, **parallel_kwargs):
    return omp_parallel(func, iterable, backend='multiprocessing', **parallel_kwargs)


def loky_parallel(func, iterable, **parallel_kwargs):
    return omp_parallel(func, iterable, backend='loky', **parallel_kwargs)


def items_wrapper(item, fn):
    """Given a function fn, that aims to accept an iterable,
    (e.g. numpy.mean), adapt it to usage with dict.items()
    where dict.items() is structured as (key: Any, values: Sequence)
    and return (key: Any, fn(values))."""
    key, values = item
    result = fn(values)
    return key, result
