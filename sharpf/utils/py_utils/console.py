from datetime import datetime
import multiprocessing
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def eprint_t(*args, **kwargs):
    ts_fmt = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f ")
    print(ts_fmt, multiprocessing.current_process().name, *args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

