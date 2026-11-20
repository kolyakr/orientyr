"""Matching backends: each one implements :class:`dronenav.types.Matcher`.

- ``rootsift``: SIFT + RootSIFT + FLANN + Lowe ratio, with the map's features cached on disk (M1-02).
"""
from dronenav.matching.rootsift import RootSiftConfig, RootSiftMatcher, rootsift

__all__ = ["RootSiftConfig", "RootSiftMatcher", "rootsift"]
