"""M2-03 · Quality gate: the contract of dronenav.quality.

The gate decides, before any matching, which frames are worth matching and which pixels of them:

- ``blur_score``: variance of the Laplacian, relative to the frame's own contrast. It falls as the frame gets
  more smeared and does not care about exposure;
- ``RunningMedian``: a frame is smeared when its score is far below the median of the frames just before it.
  There is no absolute threshold, so a scene with little texture is not thrown away as a whole;
- ``glare_mask``: saturated blobs without texture (a lamp reflected in the screen), for the matcher to skip;
- ``damp_moire``: a light Gaussian against the screen's pixel grid, and an optional notch on the grid's peak
  in the spectrum;
- ``QualityGate``: a FrameSource around any FrameSource that passes the good frames on untouched and logs every
  rejected one with a stable reason;
- ``scripts/quality_sheet.py``: the gate over a folder, a video or a flight, drawn as a contact sheet for the
  human check, with a text summary (and, on a flight with its map, the gate's verdicts against the truth).

Ground truth is synthetic and seeded: frames are patches of tests/synth.make_map, smeared, burnt or covered
with a grid by a known amount.
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from dronenav.quality import (
    REASONS,
    FrameQuality,
    QualityConfig,
    QualityGate,
    RunningMedian,
    blur_score,
    damp_moire,
    glare_mask,
)
from tests.synth import make_map
from tests.test_sim import fixture_flight  # noqa: F401 - the CI fixture laid out as a flight folder

from dronenav.sim import Blur, Glare
from dronenav.types import Frame, FrameSource

ROOT = Path(__file__).resolve().parents[1]
MAP = make_map()                                              # 1200 x 900, textured at every scale
SMOOTH = cv2.GaussianBlur(MAP, (0, 0), 4.0)                   # the same scene with little texture (fog, fields)
W, H = 320, 240


def patch(n: int, src: np.ndarray = MAP) -> np.ndarray:
    """The n-th of a seeded series of 320 x 240 patches of ``src``: a camera moving over the scene."""
    rng = np.random.default_rng(1000 + n)
    x, y = int(rng.integers(0, src.shape[1] - W)), int(rng.integers(0, src.shape[0] - H))
    return np.ascontiguousarray(src[y:y + H, x:x + W])


def smear(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


class Listed:
    """Frames held in memory, as a source with the attributes a real one has."""

    fps = 12.0

    def __init__(self, images: list[np.ndarray]):
        self.frames = [Frame(idx=3 * i, t=i / 4.0, image=im, source=f"mem#{i}", heading_prior_deg=10.0 + i,
                             scale_prior=0.5) for i, im in enumerate(images)]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def truth(self, idx: int) -> str:
        return f"truth of {idx}"


def sharp_stream(n: int = 12) -> list[np.ndarray]:
    return [patch(i) for i in range(n)]


# --- blur score --------------------------------------------------------------------------------------------------

def test_blur_score_falls_monotonically_with_gaussian_blur():
    img = patch(0)
    scores = [blur_score(img)] + [blur_score(smear(img, s)) for s in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert all(a > b for a, b in itertools.pairwise(scores)), scores
    # sigma 2 px on a 240 px frame is plainly smeared to the eye: the score must fall by far more than the
    # spread between sharp frames of one scene, or no threshold could separate them
    assert scores[3] < 0.25 * scores[0]


def test_blur_score_falls_monotonically_with_the_chaos_blur():
    # defocus + motion smear, the degradation the stress tests use
    img = patch(1)
    scores = [blur_score(Blur(s, seed=3).image(img, idx=0)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a > b for a, b in itertools.pairwise(scores)), scores


def test_blur_score_ignores_exposure():
    # an exposure swing is not blur. Half the light on the same pixels must leave the score where it was:
    # 15 % allows for the rounding of dark pixels to whole grey levels
    img = patch(2)
    dark = (img * 0.5).astype(np.uint8)
    assert blur_score(dark) == pytest.approx(blur_score(img), rel=0.15)


def test_blur_score_is_a_plain_finite_float_and_reads_grey_too():
    img = patch(3)
    s = blur_score(img)
    assert isinstance(s, float) and np.isfinite(s) and s > 0
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert blur_score(grey) == pytest.approx(s, rel=1e-6)


@pytest.mark.parametrize("image", [
    np.zeros((H, W, 3), np.uint8),                 # black-out
    np.full((H, W, 3), 255, np.uint8),             # burnt out
    np.full((H, W, 3), 128, np.uint8),             # flat grey
    np.full((1, 1, 3), 7, np.uint8),               # one pixel
    np.full((2, 3, 3), 7, np.uint8),               # smaller than the Laplacian kernel
])
def test_blur_score_of_a_blank_or_tiny_frame_is_zero_not_an_error(image):
    assert blur_score(image) == 0.0


@pytest.mark.parametrize("image", [
    np.zeros((H, W, 3), np.float32),               # wrong dtype
    np.zeros((H, W, 3), np.uint16),
    np.zeros((H, W, 4), np.uint8),                 # BGRA
    np.zeros((H,), np.uint8),                      # not an image
    np.zeros((0, W, 3), np.uint8),                 # empty
    [[1, 2], [3, 4]],                              # not an array
])
def test_functions_refuse_what_is_not_a_uint8_image(image):
    for fn in (blur_score, glare_mask, damp_moire):
        with pytest.raises(ValueError):
            fn(image)


# --- the running-median rule ---------------------------------------------------------------------------------------

def test_running_median_rejects_a_smeared_score_among_sharp_ones():
    rng = np.random.default_rng(0)
    scores = list(1.0 + 0.15 * rng.uniform(-1, 1, 30))       # sharp frames of one scene: +-15 %
    scores[17] = 0.08                                        # one smeared frame
    rule = RunningMedian()
    kept = [rule.judge(s).kept for s in scores]
    assert kept == [i != 17 for i in range(30)]


def test_running_median_reports_the_median_and_threshold_it_used():
    rule = RunningMedian(window=5, ratio=0.5, min_history=3)
    for s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        rule.judge(s)
    v = rule.judge(1.9)                                      # the last five are 2..6: median 4, threshold 2
    assert (v.median, v.threshold, v.kept) == (4.0, 2.0, False)
    v = rule.judge(1.9)                                      # now 3, 4, 5, 6, 1.9: median 4 still
    assert (v.median, v.threshold, v.kept) == (4.0, 2.0, False)


def test_first_frames_pass_until_the_median_exists():
    rule = RunningMedian(window=9, ratio=0.5, min_history=3)
    first = [rule.judge(s) for s in (1.0, 0.01, 1.0)]        # nothing to compare with yet: the benefit of the doubt
    assert all(v.kept and v.median is None and v.threshold is None for v in first)
    assert rule.judge(0.01).kept is False                     # three scores seen: the rule is on


def test_running_median_adapts_to_a_low_texture_scene():
    # the drone flies from a town (score ~1) over a field (score ~0.05): 20 times lower, like a smeared frame.
    # A fixed threshold would reject the whole field. The running median may doubt the first field frames, but
    # must have followed the scene by the time the window holds mostly field: all of the last ten are kept
    rng = np.random.default_rng(1)
    town = list(1.0 + 0.1 * rng.uniform(-1, 1, 20))
    field = list(0.05 * (1.0 + 0.1 * rng.uniform(-1, 1, 20)))
    rule = RunningMedian(window=9)
    kept = [rule.judge(s).kept for s in town + field]
    assert all(kept[:20])
    assert all(kept[-10:])
    # and a flight that is low-texture from its first frame loses nothing at all
    rule = RunningMedian(window=9)
    assert all(rule.judge(s).kept for s in field)


def test_running_median_reset_forgets_the_history():
    rule = RunningMedian(window=5, min_history=2)
    for s in (1.0, 1.0, 1.0):
        rule.judge(s)
    assert rule.judge(0.1).kept is False
    rule.reset()
    assert rule.judge(0.1).kept is True                      # a new stream: no median yet


@pytest.mark.parametrize("kw", [{"window": 0}, {"window": 2.5}, {"ratio": 0.0}, {"ratio": 1.0}, {"ratio": -0.2},
                                {"min_history": 0}, {"window": 3, "min_history": 4}])
def test_running_median_checks_its_settings(kw):
    with pytest.raises(ValueError):
        RunningMedian(**kw)


@pytest.mark.parametrize("score", [-1.0, float("nan"), float("inf")])
def test_running_median_refuses_a_score_that_is_not_a_score(score):
    with pytest.raises(ValueError):
        RunningMedian().judge(score)


# --- glare mask ------------------------------------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float((a & b).sum() / max(1, (a | b).sum()))


def _disc(cx: float, cy: float, r: float) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    return np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) <= r


def test_glare_mask_covers_a_saturated_blob():
    img = patch(4).copy()
    blob = _disc(200, 110, 45)
    img[blob] = 255
    mask = glare_mask(img)
    assert mask.dtype == np.uint8 and mask.shape == (H, W)
    assert set(np.unique(mask)) <= {0, 255}
    # the mask is the blob plus a thin safety margin (features on the blob's rim are damaged too): for a 45 px
    # disc a margin of a few px costs up to ~0.2 of IoU, so 0.75 means "this blob, about this size"
    assert _iou(mask, blob) >= 0.75
    assert (mask[blob] > 0).mean() >= 0.99                   # and nothing of the blob is left for the matcher


def test_glare_mask_covers_the_burnt_core_of_the_chaos_glare_and_stays_inside_the_spot():
    g = Glare(1.0, seed=2)
    img = g.image(patch(5), idx=0)
    cx, cy = g.centre(img.shape)
    radius = 0.45 * H                                        # chaos.GLARE_RADIUS x the shorter side
    mask = glare_mask(img)
    core = _disc(cx, cy, 0.5 * radius)                       # burnt to 255 everywhere: must be masked
    spot = _disc(cx, cy, radius)                             # beyond it the frame is untouched
    assert (mask[core] > 0).mean() >= 0.99
    assert (mask[~spot] > 0).mean() <= 0.01


def test_glare_mask_leaves_a_bright_textured_area_alone():
    # a bright roof with tiles, a sunlit car park: values up to 255, but full of detail the matcher can use
    rng = np.random.default_rng(5)
    img = patch(6).copy()
    img[60:180, 80:240] = rng.integers(200, 256, (120, 160, 1), dtype=np.uint8)
    assert glare_mask(img)[60:180, 80:240].mean() / 255 <= 0.01
    stripes = patch(6).copy()
    stripes[60:180, 80:240] = np.where((np.arange(160) // 2) % 2, 255, 190).astype(np.uint8)[None, :, None]
    assert glare_mask(stripes)[60:180, 80:240].mean() / 255 <= 0.01


def test_glare_mask_is_empty_without_glare_and_ignores_specks():
    img = patch(7).copy()
    assert not glare_mask(img).any()
    rng = np.random.default_rng(6)
    for _ in range(40):                                      # hot pixels and tiny highlights: 3 x 3 px each
        x, y = int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2))
        img[y - 1:y + 2, x - 1:x + 2] = 255
    assert not glare_mask(img).any()


def test_glare_mask_of_a_burnt_out_frame_is_everything_and_of_a_black_one_nothing():
    assert glare_mask(np.full((H, W, 3), 255, np.uint8)).all()
    assert not glare_mask(np.zeros((H, W, 3), np.uint8)).any()
    assert glare_mask(np.full((2, 2, 3), 7, np.uint8)).shape == (2, 2)      # tiny: no error


def test_blur_score_under_a_mask_ignores_the_glare():
    # a burnt blob is flat, so it would pull the score of a sharp frame down and get it rejected as "blur".
    # Scored on the usable pixels only, the frame keeps the score of the scene around the blob. The two are not
    # equal (the blob hides a quarter of the texture, and patches of one scene differ by tens of percent), so the
    # tolerance is a factor of 1.5 either way, far from the factor of 4+ that a smear gives
    clean = patch(8)
    img = clean.copy()
    img[_disc(160, 120, 70)] = 255
    usable = cv2.bitwise_not(glare_mask(img))
    assert blur_score(clean) / 1.5 <= blur_score(img, mask=usable) <= blur_score(clean) * 1.5
    with pytest.raises(ValueError):
        blur_score(img, mask=usable[:10, :10])                # a mask of another size


# --- moiré damping ---------------------------------------------------------------------------------------------------

def _grid(period: float, angle_deg: float = 0.0, amplitude: float = 25.0, size: int = 256) -> np.ndarray:
    """A screen's pixel grid seen by the camera: a sinusoid of ``period`` px along ``angle_deg``."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    a = np.deg2rad(angle_deg)
    return amplitude * np.sin(2 * np.pi * (xs * np.cos(a) + ys * np.sin(a)) / period)


def _scene(size: int = 256) -> np.ndarray:
    """Mid-grey textured scene with headroom, so adding the grid does not clip."""
    return (64 + 0.5 * MAP[100:100 + size, 200:200 + size].astype(np.float64))


def _with_grid(period: float, angle_deg: float = 0.0) -> np.ndarray:
    g = _grid(period, angle_deg)[:, :, None]
    return np.clip(_scene() + g, 0, 255).round().astype(np.uint8)


def _spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of the grey image and the frequency (cycles / px) of every bin."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    power = np.abs(np.fft.fft2(grey - grey.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(grey.shape[0]), np.fft.fftfreq(grey.shape[1]), indexing="ij")
    return power, np.stack([fx, fy])


def _peak_energy(image: np.ndarray, period: float, angle_deg: float = 0.0) -> float:
    """Energy within 3 bins of the grid's frequency (and its mirror): the grid, leakage included."""
    power, (fx, fy) = _spectrum(image)
    a = np.deg2rad(angle_deg)
    px, py = np.cos(a) / period, np.sin(a) / period
    r = 3.0 / image.shape[0]
    near = (np.hypot(fx - px, fy - py) <= r) | (np.hypot(fx + px, fy + py) <= r)
    return float(power[near].sum())


def _low_energy(image: np.ndarray, below: float = 1 / 32) -> float:
    """Energy of the content coarser than ``1 / below`` px: what a downscaled frame is matched on."""
    power, (fx, fy) = _spectrum(image)
    return float(power[np.hypot(fx, fy) < below].sum())


def test_light_gaussian_damps_a_fine_screen_grid_and_keeps_the_coarse_content():
    img = _with_grid(period=3.0)                             # a grid near the camera's pixel pitch
    out = damp_moire(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    # a Gaussian of sigma 1 px passes exp(-2 pi^2 / 9) = 11 % of the amplitude at a 3 px period, about 1 % of
    # the energy; 10 % leaves room for rounding to uint8 and for the scene's own energy at that frequency
    assert _peak_energy(out, 3.0) <= 0.10 * _peak_energy(img, 3.0)
    # content coarser than 32 px loses under 2 % of its amplitude to that Gaussian: within 5 % of its energy
    assert _low_energy(out) == pytest.approx(_low_energy(img), rel=0.05)


def test_damping_grows_with_sigma_and_sigma_zero_is_the_identity():
    img = _with_grid(period=4.0)
    energies = [_peak_energy(damp_moire(img, sigma=s), 4.0) for s in (0.0, 0.5, 1.0, 1.5)]
    assert all(a > b for a, b in itertools.pairwise(energies)), energies
    assert damp_moire(img, sigma=0.0) is img                 # nothing to do: the image itself, no copy
    with pytest.raises(ValueError):
        damp_moire(img, sigma=-1.0)


def test_damping_does_not_modify_its_input():
    img = _with_grid(period=3.0)
    before = img.copy()
    damp_moire(img, notch=True)
    assert np.array_equal(img, before)


@pytest.mark.parametrize("period, angle", [(8.0, 0.0), (6.7, 20.0), (9.3, -65.0)])
def test_notch_removes_a_coarse_grid_the_gaussian_cannot(period, angle):
    # a moiré beat is coarser than the pixel grid: at 7-9 px a light Gaussian keeps most of it (so the first
    # assert), and a stronger one would blur the scene. The notch cuts the peak itself: 10 % of the energy left
    # allows for the leakage of a grid that does not fall on a whole FFT bin
    img = _with_grid(period, angle)
    e0 = _peak_energy(img, period, angle)
    assert _peak_energy(damp_moire(img), period, angle) > 0.3 * e0
    out = damp_moire(img, notch=True)
    assert _peak_energy(out, period, angle) <= 0.10 * e0
    assert _low_energy(out) == pytest.approx(_low_energy(img), rel=0.05)


def test_notch_is_optional_and_off_by_default():
    img = _with_grid(period=8.0)
    assert np.array_equal(damp_moire(img), damp_moire(img, notch=False))
    assert not np.array_equal(damp_moire(img), damp_moire(img, notch=True))


def test_notch_leaves_a_frame_without_a_grid_alone():
    # no peak stands out of a natural scene's spectrum, so the notch has nothing to cut: the result is the
    # Gaussian-only one to within 1 grey level on average (rounding, and at most a few weak bins)
    img = np.clip(_scene(), 0, 255).round().astype(np.uint8)
    a, b = damp_moire(img, notch=True).astype(np.int16), damp_moire(img).astype(np.int16)
    assert np.abs(a - b).mean() <= 1.0


# --- the gate as a source ----------------------------------------------------------------------------------------------

def test_gate_is_a_frame_source_and_passes_good_frames_through_untouched():
    src = Listed(sharp_stream())
    gate = QualityGate(src)
    assert isinstance(gate, FrameSource)
    out = list(gate)
    assert len(out) == len(src) == len(gate)
    for got, want in zip(out, src.frames):
        assert got is want                                    # the very Frame: idx, t, source, priors and pixels


def test_gate_drops_the_smeared_frame_and_keeps_the_sharp_ones():
    images = sharp_stream()
    images[7] = smear(images[7])
    src = Listed(images)
    gate = QualityGate(src)
    assert [f.idx for f in gate] == [f.idx for i, f in enumerate(src.frames) if i != 7]
    assert len(gate) == 11
    (bad,) = gate.rejected
    assert isinstance(bad, FrameQuality)
    assert (bad.idx, bad.reason, bad.kept) == (src.frames[7].idx, "blur", False)
    assert bad.score < bad.threshold < bad.median
    assert (bad.t, bad.source) == (src.frames[7].t, "mem#7")


def test_gate_keeps_a_whole_low_texture_flight():
    # every frame here scores like a smeared frame of the textured scene; none is smeared
    images = [patch(i, SMOOTH) for i in range(12)]
    assert max(blur_score(im) for im in images) < 0.25 * min(blur_score(im) for im in sharp_stream())
    gate = QualityGate(Listed(images))
    assert len(list(gate)) == 12 and gate.rejected == []


def test_gate_gives_a_verdict_for_every_frame_in_order():
    images = sharp_stream()
    images[5] = smear(images[5])
    src = Listed(images)
    gate = QualityGate(src)
    pairs = list(gate.judged())
    assert [f for f, _ in pairs] == src.frames                # all frames, the rejected one too
    assert [q.idx for _, q in pairs] == [f.idx for f in src.frames]
    assert [q.kept for _, q in pairs] == [i != 5 for i in range(12)]
    assert all(q.reason is None for _, q in pairs if q.kept)
    assert all(q.reason in REASONS for _, q in pairs if not q.kept)
    assert [dataclasses.replace(q, mask=None) for _, q in pairs] == gate.verdicts
    # before the median exists a frame is kept without a threshold; after, every verdict names the one it met
    assert pairs[0][1].median is None and pairs[0][1].threshold is None
    assert all(q.threshold is not None and q.score >= q.threshold for _, q in pairs[4:] if q.kept)


def test_gate_is_deterministic_over_passes():
    images = sharp_stream()
    images[4], images[9] = smear(images[4]), smear(images[9])
    gate = QualityGate(Listed(images))
    first = [f.idx for f in gate]
    verdicts = list(gate.verdicts)
    assert [f.idx for f in gate] == first                     # the median starts afresh on every pass
    assert gate.verdicts == verdicts
    assert [f.idx for f in QualityGate(Listed(images))] == first


REJECT_LINE = re.compile(r"^rejected idx=(?P<idx>\d+) reason=(?P<reason>[a-z_]+) score=(?P<score>[-+.e\d]+) "
                         r"threshold=(?P<threshold>[-+.e\d]+) source=(?P<source>.+)$")


def test_gate_logs_each_rejected_frame_once_with_its_reason(caplog):
    images = sharp_stream()
    images[4], images[9] = smear(images[4]), np.zeros((H, W, 3), np.uint8)
    src = Listed(images)
    gate = QualityGate(src)
    with caplog.at_level(logging.INFO, logger="dronenav.quality"):
        list(gate)
    lines = [REJECT_LINE.match(r.getMessage()) for r in caplog.records if r.name == "dronenav.quality"]
    lines = [m for m in lines if m]
    assert [(int(m["idx"]), m["reason"], m["source"]) for m in lines] == [(src.frames[4].idx, "blur", "mem#4"),
                                                                           (src.frames[9].idx, "blank", "mem#9")]
    blur = lines[0]
    q = gate.rejected[0]
    # the numbers in the line are the ones the decision was made on
    assert float(blur["score"]) == pytest.approx(q.score, rel=1e-3)
    assert float(blur["threshold"]) == pytest.approx(q.threshold, rel=1e-3)
    assert float(blur["score"]) < float(blur["threshold"])
    # and the pass ends with one summary line that counts the reasons
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("quality gate:")]
    assert len(summary) == 1 and "kept 10 of 12" in summary[0] and "blur=1" in summary[0] and "blank=1" in summary[0]


def test_gate_rejects_a_frame_that_is_mostly_glare_and_masks_a_small_glare():
    images = sharp_stream()
    burnt = images[6].copy()
    burnt[_disc(160, 120, 150)] = 255                         # the lamp fills most of the frame
    images[6] = burnt
    spot = images[8].copy()
    blob = _disc(100, 100, 40)
    spot[blob] = 255                                          # a reflection: 7 % of the frame
    images[8] = spot
    src = Listed(images)
    gate = QualityGate(src)
    seen = {}
    for frame in gate:
        seen[frame.idx] = gate.last                           # the verdict of the frame just yielded
    assert [(q.idx, q.reason) for q in gate.rejected] == [(src.frames[6].idx, "glare")]
    assert gate.rejected[0].glare > 0.5
    q = seen[src.frames[8].idx]
    assert q.kept and q.idx == src.frames[8].idx
    assert 0.05 < q.glare < 0.15
    # the mask follows the Matcher convention: uint8, the frame's size, nonzero where pixels may be used
    assert q.mask.dtype == np.uint8 and q.mask.shape == (H, W)
    assert not q.mask[blob].any() and (q.mask[~_disc(100, 100, 55)] > 0).all()
    assert seen[src.frames[0].idx].mask is None               # no glare: no mask to carry around
    assert all(v.mask is None for v in gate.verdicts)         # the history keeps numbers, not pixels


def test_gate_rejects_blank_frames_even_before_the_median_exists():
    black, grey = np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 128, np.uint8)
    images = [black, grey] + sharp_stream(6) + [black] + sharp_stream(3)
    gate = QualityGate(Listed(images))
    kept = [q.kept for _, q in gate.judged()]
    assert kept == [False, False] + [True] * 6 + [False] + [True] * 3
    assert [q.reason for q in gate.rejected] == ["blank"] * 3
    # blank frames say nothing about the scene: they do not count towards the median
    assert gate.verdicts[4].median is None                    # only two scene frames before it


def test_gate_rejects_a_frame_too_small_to_judge():
    images = sharp_stream(5) + [np.ascontiguousarray(patch(0)[:8, :8])]
    gate = QualityGate(Listed(images))
    assert len(list(gate)) == 5
    assert [q.reason for q in gate.rejected] == ["too_small"]


def test_gate_survives_a_stream_where_every_frame_is_rejected(caplog):
    gate = QualityGate(Listed([np.zeros((H, W, 3), np.uint8)] * 5))
    with caplog.at_level(logging.INFO, logger="dronenav.quality"):
        assert list(gate) == []
    assert len(gate) == 0 and len(gate.rejected) == 5
    assert sum(bool(REJECT_LINE.match(r.getMessage())) for r in caplog.records) == 5
    assert QualityGate(Listed([])).rejected == [] and list(QualityGate(Listed([]))) == []


def test_gate_applies_the_moire_damping_only_when_asked():
    images = sharp_stream(6)
    src = Listed(images)
    gate = QualityGate(src, QualityConfig(moire_sigma=1.0))
    out = list(gate)
    assert len(out) == 6
    for got, want in zip(out, src.frames):
        assert np.array_equal(got.image, damp_moire(want.image, sigma=1.0))
        assert dataclasses.replace(got, image=want.image).__dict__ == want.__dict__    # everything else intact
    # the verdict is made on the frame as it came, so the damping cannot hide a smear from the gate
    plain = QualityGate(src)
    list(plain)
    assert [q.score for q in gate.verdicts] == [q.score for q in plain.verdicts]


def test_gate_forwards_the_wrapped_source():
    src = Listed(sharp_stream(4))
    gate = QualityGate(src)
    assert gate.source is src
    assert gate.fps == 12.0 and gate.truth(3) == "truth of 3"
    with pytest.raises(AttributeError):
        _ = gate.no_such_thing
    with pytest.raises(AttributeError):
        _ = gate._private


def test_gate_stacks_on_the_chaos_layer():
    from dronenav.sim import ChaosSource

    src = Listed(sharp_stream())
    # strength 1 for every frame: the whole stream is equally smeared, so relative to itself nothing stands out
    gate = QualityGate(ChaosSource(src, [Blur(1.0, seed=1)]))
    assert len(list(gate)) == 12


@pytest.mark.parametrize("kw", [{"window": 0}, {"blur_ratio": 1.5}, {"min_history": 0}, {"max_glare": 0.0},
                                {"max_glare": 1.5}, {"min_side": 0}, {"moire_sigma": -1.0}, {"glare_level": 300},
                                {"blank_std": -1.0}])
def test_config_checks_its_values(kw):
    with pytest.raises(ValueError):
        QualityConfig(**kw)


def test_reasons_are_the_stable_vocabulary():
    assert REASONS == ("too_small", "blank", "glare", "blur")


# --- scripts/quality_sheet.py ------------------------------------------------------------------------------------

def _sheet(*args) -> subprocess.CompletedProcess:
    """Run scripts/quality_sheet.py as a user would; the caller checks the exit code."""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "quality_sheet.py"), *map(str, args)],
                          capture_output=True, text=True, check=False, cwd=ROOT)


def _check_sheet(out: Path, pages: int = 1) -> str:
    assert sorted(p.name for p in out.iterdir()) == [*(f"sheet_{n:02d}.png" for n in range(1, pages + 1)),
                                                     "summary.txt"]
    for n in range(1, pages + 1):
        img = cv2.imread(str(out / f"sheet_{n:02d}.png"))
        assert img is not None
        # thumbnails in rows: wide enough to judge sharpness, small enough to open anywhere
        assert 600 <= img.shape[1] <= 4000 and 150 <= img.shape[0] <= 6000, img.shape
        assert img.std() > 10                                 # something is drawn on it
    return (out / "summary.txt").read_text()


def _frames_folder(folder: Path, n: int = 10, smeared: int = 6) -> Path:
    folder.mkdir()
    for i in range(n):
        assert cv2.imwrite(str(folder / f"frame_{i:03d}.png"), smear(patch(i)) if i == smeared else patch(i))
    return folder


def test_sheet_of_a_frames_folder(tmp_path):
    frames = _frames_folder(tmp_path / "frames")
    r = _sheet("--frames", frames, "--out", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    summary = _check_sheet(tmp_path / "out")
    rows = [ln.split() for ln in summary.splitlines() if ln.split() and ln.split()[0].isdigit()]
    assert [int(row[0]) for row in rows] == list(range(10))  # one line per frame, by idx
    assert [row[-1] for row in rows] == ["blur" if i == 6 else "kept" for i in range(10)]
    assert "kept 9 of 10" in summary and "blur 1" in summary
    assert "kept 9 of 10" in r.stdout
    assert str(tmp_path / "out" / "sheet_01.png") in r.stdout and str(tmp_path / "out" / "summary.txt") in r.stdout
    assert "rejected idx=6 reason=blur" in r.stderr           # the gate's own log line, on stderr


def test_sheet_pages_a_long_source(tmp_path):
    frames = _frames_folder(tmp_path / "frames", n=10)
    r = _sheet("--frames", frames, "--out", tmp_path / "out", "--per-page", 4)
    assert r.returncode == 0, r.stderr
    _check_sheet(tmp_path / "out", pages=3)


def test_sheet_of_a_video(tmp_path):
    path = tmp_path / "take.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert out.isOpened()
    for i in range(12):
        out.write(smear(patch(i), 4.0) if i == 8 else patch(i))
    out.release()
    r = _sheet("--video", path, "--every", 2, "--out", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    summary = _check_sheet(tmp_path / "out")
    assert "kept 5 of 6" in summary                           # frames 0, 2, .., 10; frame 8 is the smeared one


def test_sheet_of_a_flight_compares_the_gate_with_the_truth(tmp_path, fixture_flight):  # noqa: F811
    flight, crop = fixture_flight
    map_path = tmp_path / "map.png"
    assert cv2.imwrite(str(map_path), crop)
    r = _sheet("--flight", flight, "--keyframes", "--map", map_path, "--cache", tmp_path / "cache",
               "--out", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    summary = _check_sheet(tmp_path / "out")
    # three sharp keyframes of the mild flight: all kept, all found on their own map within the tolerance
    assert "kept 3 of 3" in summary
    table = summary[summary.index("gate against the truth"):]
    kept = next(ln for ln in table.splitlines() if ln.split()[:1] == ["kept"]).split()
    dropped = next(ln for ln in table.splitlines() if ln.split()[:1] == ["dropped"]).split()
    assert kept[1:4] == ["3", "3", "0"]                       # frames, fix right, fix wrong or missing
    assert dropped[1:4] == ["0", "0", "0"]
    assert "gate against the truth" in r.stdout


@pytest.mark.parametrize("args, message", [
    ((), "one of"),                                           # no source
    (("--frames", "a", "--video", "b"), "one of"),            # two sources
    (("--frames", "no_such_folder"), "no frames folder"),
    (("--video", "no_such.mp4"), "no video"),
    (("--frames", "FRAMES", "--map", "m.png"), "--map"),      # truth exists for a flight only
    (("--frames", "FRAMES", "--per-page", 0), "--per-page"),
])
def test_sheet_refuses_bad_arguments_without_a_traceback(tmp_path, args, message):
    frames = _frames_folder(tmp_path / "frames", n=3, smeared=-1)
    args = [frames if a == "FRAMES" else a for a in args]
    r = _sheet(*args, "--out", tmp_path / "out")
    assert r.returncode == 2 and message in r.stderr and "Traceback" not in r.stderr, r.stderr
    assert not (tmp_path / "out").exists()
