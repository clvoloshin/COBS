import threading

class LockedIterator(object):
    def __init__(self, it):
        self._lock = threading.Lock()
        self._it = it.__iter__()
        if hasattr(self._it, 'close'):
            def close(self):
                with self._lock:
                    self._it.close()
            self.__setattr__('close', close)

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return LockedIterator(f(*a, **kw))
    return g

