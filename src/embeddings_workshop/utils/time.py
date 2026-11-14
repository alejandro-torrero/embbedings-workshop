import datetime
import time


def timed_function(f):
    def inner(*args, **kwargs):
        print(f"Starting timed function")
        start = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"Timed process finished. {f.__name__} took {elapsed:.3f}s")

    return inner
