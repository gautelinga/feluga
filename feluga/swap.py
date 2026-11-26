"""A folder written whole, then swapped in for the one it replaces.

    new = new_folder(target)    # <target>.new
    recover(target)             # before reading target, or writing a new one
    ... write everything into new ...
    swap_in(new, target)

Rewritten in place, file by file, a folder killed between two files is a mismatched
set nothing can read. Written aside and swapped in by renaming, a kill at any moment
leaves a whole folder under one of three names: `target`, `<target>.old` (the one
replaced, renamed aside before the new one is renamed in) or `new`. `recover` puts
it back under `target`. Call both on one process only.
"""
import os
import shutil


def _old(target):
    return os.path.normpath(target) + ".old"


def new_folder(target):
    """`<target>.new`, where a new `target` is written before `swap_in` (a trailing
    slash on `target` is dropped first)."""
    return os.path.normpath(target) + ".new"


def swap_in(new, target):
    """Make the folder `new`, written whole, `target`: the previous `target` is
    renamed aside, `new` renamed in, the previous one removed."""
    old = _old(target)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(target):
        os.rename(target, old)
    os.rename(new, target)
    shutil.rmtree(old, ignore_errors=True)


def recover(target, new=None, log=None):
    """If a `swap_in` was killed between its renames, `target` is missing and
    `<target>.old` is there: put `new` (default `new_folder(target)`; whole, since
    the renames start once it is written) in place, or else the old one. Anything
    else is left as it is. Returns whether a folder was put back; `log(message)` is
    told when one is."""
    old = _old(target)
    new = new_folder(target) if new is None else new
    if os.path.exists(target) or not os.path.exists(old):
        return False
    which = "new" if os.path.exists(new) else "old"
    os.rename(new if which == "new" else old, target)
    shutil.rmtree(old, ignore_errors=True)
    if log is not None:
        log("{}: a swap was interrupted between its renames; the {} folder is put "
            "back".format(target, which))
    return True
