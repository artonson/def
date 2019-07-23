
import sys

if sys.version_info >= (3, 7):
    # namedtuple with defaults exists in python 3.7
    # https://stackoverflow.com/questions/11351032/namedtuple-and-default-values-for-optional-keyword-arguments

    from collections import namedtuple
    namedtuple_with_defaults = namedtuple

else:
    import collections


    def namedtuple_with_defaults(typename, field_names, defaults=()):
        T = collections.namedtuple(typename, field_names)
        T.__new__.__defaults__ = (None,) * len(T._fields)
        if isinstance(defaults, collections.Mapping):
            prototype = T(**defaults)
        else:
            prototype = T(*defaults)
        T.__new__.__defaults__ = tuple(prototype)
        return T
