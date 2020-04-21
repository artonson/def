import sys
from datetime import datetime


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def eprint_t(*args, **kwargs):
    ts_fmt = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f ")
    print(ts_fmt, *args, file=sys.stderr, **kwargs)

