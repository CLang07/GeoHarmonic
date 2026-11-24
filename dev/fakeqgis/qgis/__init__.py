from anything import Anything  # noqa: F401  (dev/fakeqt must be on the path)


def __getattr__(name):
    return Anything
