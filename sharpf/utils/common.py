import os
import sys
from datetime import datetime


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def eprint_t(*args, **kwargs):
    ts_fmt = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f ")
    eargs = (ts_fmt, ) + args
    eprint(eargs, **kwargs)


def change_ext(filename, new_ext):
    name, old_ext = os.path.splitext(filename)
    return name + new_ext


