from distutils.spawn import find_executable


def set_attribute(name, value):
    """Decorator that sets an attribute."""

    def decorator(func):
        setattr(func, name, value)
        return func

    return decorator
