import os
import sys
from datetime import datetime


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def eprint_t(*args, **kwargs):
    ts_fmt = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f ")
    print(ts_fmt, *args, file=sys.stderr, **kwargs)


def change_ext(filename, new_ext):
    name, old_ext = os.path.splitext(filename)
    return name + new_ext


def add_suffix(filename, suffix, suffix_sep='_'):
    name, ext = os.path.splitext(filename)
    return '{}{}{}{}'.format(name, suffix_sep, suffix, ext)
