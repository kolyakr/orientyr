"""Markers for tests that need model weights (not a test module itself).

Weights are never committed and tests never touch the network, so a test that needs trained weights runs only
where they are already in the local cache and skips cleanly elsewhere (CI). To get them once:
``python scripts/xfeat_timing.py`` (it downloads them into the torch cache), then run the tests again.
"""
import pytest

from dronenav.matching import xfeat_weights_path

needs_xfeat_weights = pytest.mark.skipif(
    not xfeat_weights_path().is_file(),
    reason=f"XFeat weights are not at {xfeat_weights_path()} (scripts/xfeat_timing.py downloads them once)")
