"""Generating domain proxies that expose only their public API,
while maintaining their characteristics mixins dependencies.
"""

import inspect
from typing import Any

import skdecide.builders
from skdecide import Domain


def get_domain_mixin_classes(obj: Any) -> tuple[type[Domain], ...]:
    """Extract scikit-decide builder mixins from an object or class.

    This helper inspects the Method Resolution Order (MRO) of the provided
    object and identifies which classes belong to the
    :mod:`skdecide.builders.domain` module.

    # Parameters
    obj: The domain instance or class to inspect.

    # Returns
    A tuple of mixin classes used by the domain.
    """
    cls = obj if inspect.isclass(obj) else type(obj)

    # Identify all valid builder mixins in the skdecide.builders.domain namespace
    builder_mixin_set = {
        o for _, o in inspect.getmembers(skdecide.builders.domain) if inspect.isclass(o)
    }

    return tuple(base for base in cls.__mro__ if base in builder_mixin_set)


def create_public_proxy(obj: Any) -> Any:
    """Create a proxy that exposes only the public API of a domain.

    The resulting proxy object inherits from the same scikit-decide mixins
    as the original object to ensure compatibility with solvers (via
    isinstance checks).

    # Parameters
    obj: The original domain instance to be wrapped.

    # Returns
    A dynamic proxy object mimicking the domain's capabilities
    while enforcing public-only access.

    # Raises
    AttributeError: If access to members starting with '_' is attempted.

    """
    mixins = get_domain_mixin_classes(obj)

    class PublicProxy:
        """Internal base for the proxy to trap attribute access.

        Use __getattribute__ instead of __getattr__ to avoid accessing methods
        from inherited characteristics mixins.

        """

        def __getattribute__(self, item: str) -> Any:
            if item.startswith("_"):
                raise AttributeError(
                    f"Access to private member '{item}' is not allowed."
                )
            return getattr(obj, item)

        def __setattr__(self, name: str, value: Any) -> None:
            if name.startswith("_"):
                raise AttributeError(
                    f"Modification of private member '{name}' is not allowed."
                )
            setattr(obj, name, value)

        def __dir__(self) -> list[str]:
            # Filter out private members for discovery/tab-completion
            return [attr for attr in dir(obj) if not attr.startswith("_")]

        def __repr__(self) -> str:
            return f"PublicProxy({repr(obj)})"

    # Construct the dynamic class
    # MRO order: Proxy public/private api filter  -> Mixin capabilities
    dynamic_name = f"PublicProxy_{type(obj).__name__}"
    ProxyClass = type(dynamic_name, (PublicProxy, *mixins), {})

    return ProxyClass()
