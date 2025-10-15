"""Math function utils."""

__all__ = ("clamp",)


def clamp(n, minn, maxn):
    """Clamp n between minn and maxn (inclusive)."""
    return max(min(maxn, n), minn)
