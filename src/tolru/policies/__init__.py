from .base import ReplacementPolicy
from .global_lru import GlobalLRU
from .token_lru import TokenOrderedLRU


def make_policy(name: str, cfg):
    if name == "global_lru":
        return GlobalLRU()
    if name == "token_lru":
        return TokenOrderedLRU(cfg)
    raise ValueError(f"unknown policy {name!r}")


__all__ = ["ReplacementPolicy", "GlobalLRU", "TokenOrderedLRU", "make_policy"]
