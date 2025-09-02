from functools import wraps
from typing import Any, Type, TypeVar

T = TypeVar("T")


def singleton(cls: Type[T]) -> Type[T]:
    """
    Class decorator to make a class a singleton.
    Ensures only one instance of the decorated class exists.

    Usage:
        @singleton
        class MyClass:
            pass

        a = MyClass()
        b = MyClass()
        assert a is b  # True
    """
    instances: dict[Type[T], T] = {}

    @wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance  # returns a callable that behaves like the class
