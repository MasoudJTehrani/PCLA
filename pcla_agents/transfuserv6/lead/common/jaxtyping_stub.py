from __future__ import annotations

from typing import Any


class _JaxtypingType:
    """Simple subscriptable placeholder returning Any for type hints."""

    def __class_getitem__(cls, _item: Any) -> Any:  # pragma: no cover - trivial stub
        return Any


class _JaxtypingStub:
    """Lightweight fallback to avoid importing jaxtyping at runtime."""

    def jaxtyped(self, *args: Any, **kwargs: Any):  # pragma: no cover - decorator stub
        def decorator(fn):
            return fn

        return decorator

    def __getattr__(self, _name: str) -> Any:
        # Return a subscriptable placeholder type for jt.UInt8, jt.Int32, jt.Bool, etc.
        return _JaxtypingType


jt = _JaxtypingStub()
