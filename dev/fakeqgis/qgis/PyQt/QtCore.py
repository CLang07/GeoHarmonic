from anything import Anything  # noqa: F401


def __getattr__(name):
    return Anything
