from functools import lru_cache, wraps
import weakref


class Handle:
    __slots__ = ('ref',)

    def __init__(self, target):
        self.ref = weakref.ref(target)


_HANDLES = weakref.WeakKeyDictionary()


def handle(target):
    h = _HANDLES.get(target)
    if h is None:
        h = _HANDLES[target] = Handle(target)
    return h


def weak_method_lru_cache(maxsize):
    def decorate(method):
        @lru_cache(maxsize=maxsize)
        def table(owner, *args, **kwargs):
            return method(owner.ref(), *args, **kwargs)

        @wraps(method)
        def memoized(self, *args, **kwargs):
            return table(handle(self), *args, **kwargs)
        memoized.cache_info, memoized.cache_clear, memoized.table = table.cache_info, table.cache_clear, table
        return memoized
    return decorate
