"""Matching backends: each one implements :class:`dronenav.types.Matcher`.

- ``rootsift``: SIFT + RootSIFT + FLANN + Lowe ratio, with the map's features cached on disk (M1-02).
- ``xfeat``: the XFeat network (via kornia) for the fast tracking path, frame tried at several turns (M4-03).
  Importing it is light: torch and kornia are imported when an ``XFeatMatcher`` is built.
- ``lightglue``: DISK (or SuperPoint) features matched by LightGlue, on MPS / CUDA / CPU (M4-02). torch and the
  weights load on first use, so importing this package stays light.
"""
from dronenav.matching.lightglue import LightGlueConfig, LightGlueMatcher
from dronenav.matching.rootsift import RootSiftConfig, RootSiftMatcher, rootsift
from dronenav.matching.xfeat import XFeatConfig, XFeatMatcher, XFeatWeightsMissing, xfeat_weights_path

__all__ = ["LightGlueConfig", "LightGlueMatcher", "RootSiftConfig", "RootSiftMatcher", "XFeatConfig", "XFeatMatcher",
           "XFeatWeightsMissing", "rootsift", "xfeat_weights_path"]
