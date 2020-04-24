import shutil
import os, os.path


def require_empty(d, recreate=False):
    if os.path.exists(d):
        if recreate:
            shutil.rmtree(d)
        else:
            raise OSError('Path {} exists and no --overwrite set. Exiting'.format(d))
    os.makedirs(d)


def change_ext(filename, new_ext):
    name, old_ext = os.path.splitext(filename)
    return name + new_ext


def add_suffix(filename, suffix, suffix_sep='_'):
    name, ext = os.path.splitext(filename)
    return '{}{}{}{}'.format(name, suffix_sep, suffix, ext)
