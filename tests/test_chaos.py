"""M6-02 · Chaos: composable degradations with a strength and a seed, applied on top of any frame source.

Done when (issue #54 and its spec update): each degradation has a visual check sheet and a seed; chaos is applied
at run time on top of FlightSource frames and on the cut-out generator's frames. Encoded here:

1. strength 0 is the identity, exactly, for every degradation;
2. the effect grows with the strength, by a proxy measured per degradation (sharpness, error, brightness, area,
   count, window length, angle), with an explicit tolerance;
3. the same seed gives identical output, another seed another one; randomness depends on (seed, frame idx), never
   on the order of iteration;
4. images keep their shape and dtype, and the input is never modified in place;
5. degradations compose in the order given;
6. ChaosSource keeps the source contract: Frames, len, idx and t (dropped frames leave gaps), truth by idx;
7. a strength outside [0, 1] or a bad seed is refused;
8. scripts/chaos_sheet.py writes one check sheet per degradation, with no data/ at all and on a flight folder.
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.sim import CutoutSource, FlightSource, cutout_route
from dronenav.sim.chaos import (
    DEGRADATIONS,
    Blur,
    ChaosSource,
    CompassError,
    Drop,
    Gamma,
    Glare,
    Jam,
    Jpeg,
    Noise,
    Occlusion,
    degrade,
)
from dronenav.types import Frame, FrameSource
from tests.synth import make_map
from tests.test_sim import fixture_flight  # noqa: F401 - the CI fixture laid out as a flight folder

ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
IMAGE = (Blur, Noise, Jpeg, Gamma, Glare, Occlusion)
STREAM = (Drop, Jam, CompassError)
NAMES = ("blur", "noise", "jpeg", "gamma", "glare", "occlusion", "drop", "jam", "compass")


@pytest.fixture(scope="module")
def image():
    """A 320 x 240 textured frame with no pure black or white pixels, so covered (black) and saturated (white)
    areas can be counted exactly."""
    return np.clip(make_map(320, 240, seed=1, shapes=70), 8, 240).astype(np.uint8)


@pytest.fixture(scope="module")
def cutouts():
    """40 cut-out frames at 2 fps (20 s) with perfect priors: the cut-out generator the chaos must sit on."""
    world = make_map(900, 700, seed=2, shapes=500)
    poses = cutout_route((900, 700), 40, seed=4, frame_size=(96, 72))
    return CutoutSource(world, poses, (96, 72), fps=2.0, priors=True)


def _frame(image, idx=0, heading=None):
    return Frame(idx=idx, t=float(idx), image=image, source=f"test#{idx}", heading_prior_deg=heading,
                 scale_prior=0.5)


def _mse(a, b):
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2))


def _same(a: Frame, b: Frame) -> bool:
    return ((a.idx, a.t, a.source, a.heading_prior_deg, a.scale_prior)
            == (b.idx, b.t, b.source, b.heading_prior_deg, b.scale_prior) and np.array_equal(a.image, b.image))


def _angle(a, b):
    """Smallest absolute difference of two headings, degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


# --- 1. strength 0 is the identity -----------------------------------------------------------------------------

def test_every_named_degradation_exists():
    assert tuple(DEGRADATIONS) == NAMES
    assert [DEGRADATIONS[n] for n in NAMES] == [*IMAGE, *STREAM]
    assert all(DEGRADATIONS[n].name == n for n in NAMES)


@pytest.mark.parametrize("cls", IMAGE)
def test_strength_zero_leaves_the_pixels_alone(cls, image):
    assert np.array_equal(cls(0.0, seed=3).image(image, idx=5), image)


@pytest.mark.parametrize("cls", [*IMAGE, *STREAM])
def test_strength_zero_leaves_the_stream_alone(cls, cutouts):
    plain = list(cutouts)
    out = list(ChaosSource(cutouts, [cls(0.0, seed=3)]))
    assert len(out) == len(plain) and all(_same(a, b) for a, b in zip(out, plain, strict=True))


# --- 2. the effect grows with the strength ---------------------------------------------------------------------

def _steps(values):
    return list(zip(values, values[1:], strict=False))


def test_blur_removes_detail(image):
    sharp = [cv2.Laplacian(cv2.cvtColor(Blur(s, seed=1).image(image, 0), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
             for s in STRENGTHS]
    # the variance of the Laplacian is the usual sharpness measure; each quarter step must lose at least 5 % of
    # it: far above rounding noise, and loose enough not to pin the kernel
    assert all(b < 0.95 * a for a, b in _steps(sharp)), sharp
    assert sharp[-1] < 0.05 * sharp[0]                     # full strength: a frame nobody can match


def test_noise_adds_error(image):
    mse = [_mse(Noise(s, seed=1).image(image, 0), image) for s in STRENGTHS]
    # MSE grows with sigma squared (1 : 4 : 9 : 16), a little less where pixels clip; 1.2x per step is safe
    assert mse[0] == 0 and all(b > 1.2 * a for a, b in _steps(mse[1:])), mse
    assert mse[1] > 20                                     # visible already at a quarter (sigma of several levels)


def test_jpeg_adds_error(image):
    mse = [_mse(Jpeg(s, seed=1).image(image, 0), image) for s in STRENGTHS]
    # lower quality, coarser quantisation: the error rises every step; 5 % margin, the low strengths are close
    assert mse[0] == 0 and mse[1] > 0 and all(b > 1.05 * a for a, b in _steps(mse[1:])), mse
    # blocks at full strength: over 3x the error of a quarter (measured 5x; this noisy texture costs JPEG ~70
    # of MSE already at quality 72, so the first draft's 10x was a guess no JPEG setting reaches)
    assert mse[-1] > 3 * mse[1]
    # and no jump at the start: a hair of strength is near-lossless (measured 14 at quality 95 with full chroma;
    # the usual 4:2:0 chroma would cost 108 on its own, more than a quarter of the Noise scale)
    assert 0 < _mse(Jpeg(0.01, seed=1).image(image, 0), image) < 25


def test_gamma_moves_the_brightness(image):
    shift = [abs(float(Gamma(s, seed=1).image(image, 0).mean()) - float(image.mean())) for s in STRENGTHS]
    # the mean moves away from the original, at least 2 grey levels more each step (uint8 rounding is < 1)
    assert shift[0] == 0 and all(b > a + 2 for a, b in _steps(shift)), shift


def test_gamma_goes_the_way_it_is_told(image):
    assert Gamma(0.5, direction="dark").image(image, 0).mean() < image.mean() - 5
    assert Gamma(0.5, direction="bright").image(image, 0).mean() > image.mean() + 5
    with pytest.raises(ValueError, match="direction"):
        Gamma(0.5, direction="sideways")


def test_glare_saturates_a_growing_spot(image):
    def white(img):
        return float((img.min(axis=2) >= 250).mean())      # fraction of the frame burnt out
    area = [white(Glare(s, seed=1).image(image, 0)) for s in STRENGTHS]
    # the fixture has no pixel above 240, so every white pixel is glare; the spot gains at least 0.5 % of the
    # frame each step (its core is ~0.75 % at a quarter) and burns out over 8 % at full strength
    assert area[0] == 0 and all(b > a + 0.005 for a, b in _steps(area)), area
    assert area[-1] > 0.08
    # a spot, not a wash: the corner farthest from it is untouched at half strength
    out = Glare(0.5, seed=1).image(image, 0)
    assert any(np.array_equal(out[c], image[c]) for c in [(0, 0), (0, -1), (-1, 0), (-1, -1)])


def test_occlusion_covers_a_growing_box(image):
    masks = [Occlusion(s, seed=1).image(image, 0).max(axis=2) == 0 for s in STRENGTHS]
    area = [float(m.mean()) for m in masks]
    # the fixture has no black pixel, so black = covered. The box covers strength x 50 % of the frame; 1 % of
    # tolerance for rounding its sides to whole pixels (about 0.3 % here)
    assert area == pytest.approx([0.5 * s for s in STRENGTHS], abs=0.01), area
    for small, big in _steps(masks):
        assert not (small & ~big).any()                    # it grows in place, it does not jump around
    ys, xs = np.nonzero(masks[2])                          # one solid rectangle
    assert masks[2][ys.min():ys.max() + 1, xs.min():xs.max() + 1].all()


def test_drops_remove_more_frames(image):
    dropped = [{i for i in range(1000) if Drop(s, seed=1).drops(i)} for s in STRENGTHS]
    rate = [len(d) / 1000 for d in dropped]
    # strength x 50 % of the frames go; +-0.05 is three sigma of a 1000-frame binomial (sigma <= 0.016)
    assert rate == pytest.approx([0.5 * s for s in STRENGTHS], abs=0.05), rate
    assert all(a < b for a, b in _steps(dropped))          # strictly more, and the same frames plus new ones


def test_dropped_frames_leave_gaps_in_the_indices(cutouts):
    drop = Drop(0.5, seed=1)
    out = list(ChaosSource(cutouts, [drop]))
    kept = [i for i in range(len(cutouts)) if not drop.drops(i)]
    assert 0 < len(out) < len(cutouts)
    assert [f.idx for f in out] == kept                    # no renumbering: idx still names the source frame
    assert [f.t for f in out] == [i / 2.0 for i in kept]   # and its time, so the truth still lines up
    plain = {f.idx: f for f in cutouts}
    assert all(_same(f, plain[f.idx]) for f in out)        # the frames that survive are untouched


def test_the_jam_window_grows(cutouts):
    plain = list(cutouts)
    lengths = []
    for s in STRENGTHS:
        jam = Jam(s, seed=1, start_s=2.0, max_s=10.0)
        out = list(ChaosSource(cutouts, [jam]))
        assert [(f.idx, f.t, f.source, f.heading_prior_deg) for f in out] == \
               [(f.idx, f.t, f.source, f.heading_prior_deg) for f in plain]      # jammed, not missing
        hit = [f.idx for f, p in zip(out, plain, strict=True) if not np.array_equal(f.image, p.image)]
        assert hit == [f.idx for f in plain if jam.covers(f.t)]
        assert hit == list(range(4, 4 + len(hit)))                               # one run, from t = 2 s (2 fps)
        assert jam.length_s == pytest.approx(10.0 * s)
        lengths.append(len(hit))
    # the window lasts strength x max_s: 0, 2.5, 5, 7.5, 10 s of a 2 fps stream, exactly
    assert lengths == [0, 5, 10, 15, 20]


def test_a_jammed_frame_is_garbage_of_the_same_shape(cutouts):
    plain = list(cutouts)
    out = list(ChaosSource(cutouts, [Jam(1.0, seed=1, start_s=2.0)]))
    a, b = out[6].image, plain[6].image
    assert a.shape == b.shape and a.dtype == np.uint8
    # nothing of the scene is left: the correlation with the real frame is noise-level (|r| < 0.2 on 6912 px)
    r = np.corrcoef(a.astype(float).ravel(), b.astype(float).ravel())[0, 1]
    assert abs(r) < 0.2 and a.std() > 10                   # and it is noise, not a blank frame


def test_a_jam_without_a_start_picks_one_from_the_seed():
    a, b = Jam(0.5, seed=1), Jam(0.5, seed=2)
    assert a.start_s == Jam(0.5, seed=1).start_s and a.start_s != b.start_s
    assert 0 <= a.start_s <= 20 and a.length_s == pytest.approx(5.0)
    assert a.covers(a.start_s) and a.covers(a.start_s + 4.99) and not a.covers(a.start_s + 5.0)
    assert not a.covers(a.start_s - 0.01)


def test_compass_error_grows_and_scales_with_the_strength():
    full = np.array([CompassError(1.0, seed=1).error_deg(i) for i in range(200)])
    mean_abs = []
    for s in STRENGTHS:
        err = np.array([CompassError(s, seed=1).error_deg(i) for i in range(200)])
        assert err == pytest.approx(s * full, abs=1e-9)    # the same error pattern, scaled: exactly monotonic
        mean_abs.append(float(np.abs(err).mean()))
    assert mean_abs[0] == 0 and all(b > a + 3 for a, b in _steps(mean_abs)), mean_abs
    # full strength: a 20 degree bias plus 5 degrees of noise; the mean |error| of 200 frames is 20 +- 1.5
    # (the noise averages to 5 / sqrt(200) = 0.35)
    assert mean_abs[-1] == pytest.approx(20.0, abs=1.5)
    assert 3 < full.std() < 7                              # the noise is there, around the bias


def test_compass_error_turns_the_heading_prior_only(image):
    comp = CompassError(1.0, seed=1)
    for idx, heading in ((0, 10.0), (1, 350.0), (2, 0.0), (3, 179.5)):
        f = _frame(image, idx, heading)
        out = comp.apply(f)
        assert 0 <= out.heading_prior_deg < 360            # wrapped, as Frame promises
        assert _angle(out.heading_prior_deg, heading) == pytest.approx(abs(comp.error_deg(idx)), abs=1e-6)
        assert out.image is f.image and out.scale_prior == f.scale_prior and (out.idx, out.t) == (f.idx, f.t)
    assert comp.apply(_frame(image, 0, None)).heading_prior_deg is None      # no compass, nothing to break


# --- 3. seeds --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", IMAGE)
def test_image_degradations_repeat_with_their_seed_and_differ_between_seeds(cls, image):
    a = cls(0.6, seed=7).image(image, idx=3)
    assert np.array_equal(a, cls(0.6, seed=7).image(image, idx=3))
    assert not np.array_equal(a, cls(0.6, seed=8).image(image, idx=3))


@pytest.mark.parametrize("cls", [Blur, Noise, Jpeg, Gamma])
def test_per_frame_effects_change_from_frame_to_frame(cls, image):
    d = cls(0.6, seed=7)
    assert not np.array_equal(d.image(image, idx=3), d.image(image, idx=4))


@pytest.mark.parametrize("cls", [Glare, Occlusion])
def test_spots_and_boxes_stay_put_along_a_stream(cls, image):
    d = cls(0.6, seed=7)                                   # dirt on the lens, a lamp on the screen: not per frame
    assert np.array_equal(d.image(image, idx=3), d.image(image, idx=4))


def test_stream_degradations_repeat_with_their_seed_and_differ_between_seeds():
    def drops(seed):
        return [Drop(0.6, seed=seed).drops(i) for i in range(200)]

    def errors(seed):
        return [CompassError(0.6, seed=seed).error_deg(i) for i in range(50)]
    assert drops(7) == drops(7) and drops(7) != drops(8)
    assert errors(7) == errors(7) and errors(7) != errors(8)
    img = np.zeros((24, 32, 3), np.uint8)
    jam7 = Jam(1.0, seed=7, start_s=0.0).apply(_frame(img, 2)).image
    assert np.array_equal(jam7, Jam(1.0, seed=7, start_s=0.0).apply(_frame(img, 2)).image)
    assert not np.array_equal(jam7, Jam(1.0, seed=8, start_s=0.0).apply(_frame(img, 2)).image)
    assert not np.array_equal(jam7, Jam(1.0, seed=7, start_s=0.0).apply(_frame(img, 3)).image)


class _Frames:
    """A bare frame source over a list: any order, any subset."""

    def __init__(self, frames):
        self.frames = list(frames)

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)


def test_randomness_depends_on_the_frame_index_not_on_the_iteration(cutouts):
    chain = [Blur(0.4, seed=1), Noise(0.5, seed=2), Jpeg(0.5, seed=3), Gamma(0.3, seed=4), Glare(0.4, seed=5),
             Occlusion(0.2, seed=6), Jam(0.3, seed=7, start_s=3.0), CompassError(0.5, seed=8), Drop(0.3, seed=9)]
    src = ChaosSource(cutouts, chain)
    first = {f.idx: f for f in src}
    again = {f.idx: f for f in src}                        # a second pass over the same source
    assert first.keys() == again.keys() and all(_same(first[i], again[i]) for i in first)
    plain = list(cutouts)
    part = {f.idx: f for f in ChaosSource(_Frames(plain[::-3]), chain)}       # backwards, every third frame
    assert part and all(_same(part[i], first[i]) for i in part)
    assert part.keys() == {f.idx for f in plain[::-3]} & first.keys()         # the same frames are dropped


# --- 4. shape, dtype, no in-place edits ------------------------------------------------------------------------

@pytest.mark.parametrize("cls", IMAGE)
@pytest.mark.parametrize("size", [(240, 320), (17, 23), (64, 48)])
def test_images_keep_their_shape_and_the_input_is_not_touched(cls, size, image):
    img = np.ascontiguousarray(image[:size[0], :size[1]])
    before = img.copy()
    for s in (0.3, 1.0):
        out = cls(s, seed=2).image(img, idx=1)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert np.array_equal(img, before)


@pytest.mark.parametrize("cls", IMAGE)
def test_image_degradations_change_only_the_pixels(cls, image):
    f = _frame(image, 4, 123.0)
    out = cls(0.7, seed=2).apply(f)
    assert isinstance(out, Frame) and not np.array_equal(out.image, f.image)
    assert dataclasses.replace(out, image=f.image).__dict__ == f.__dict__


# --- 5. composition --------------------------------------------------------------------------------------------

def test_the_chain_runs_in_the_order_given(image):
    box, noise = Occlusion(0.4, seed=1), Noise(0.6, seed=2)
    f = _frame(image, 3)
    covered = box.image(image, 3).max(axis=2) == 0
    box_then_noise = degrade(f, [box, noise])
    noise_then_box = degrade(f, [noise, box])
    assert np.array_equal(box_then_noise.image, noise.image(box.image(image, 3), 3))
    assert np.array_equal(noise_then_box.image, box.image(noise.image(image, 3), 3))
    assert (noise_then_box.image[covered] == 0).all()      # the box went on last: flat black
    assert box_then_noise.image[covered].max() > 30        # the noise went on last: the box is noisy too
    assert degrade(f, []) is f


def test_a_dropped_frame_ends_the_chain(image):
    drop = Drop(1.0, seed=1)
    idx = next(i for i in range(50) if drop.drops(i))
    assert degrade(_frame(image, idx), [drop, Noise(0.5)]) is None
    assert degrade(_frame(image, idx), [Noise(0.5), drop]) is None


def test_the_chain_must_hold_degradations(cutouts, image):
    with pytest.raises(TypeError, match="Degradation"):
        ChaosSource(cutouts, [Blur(0.5), "noise"])
    with pytest.raises(TypeError, match="Degradation"):
        degrade(_frame(image), [lambda f: f])


# --- 6. the source contract ------------------------------------------------------------------------------------

def test_chaos_source_is_a_frame_source_with_the_same_frames_and_truth(cutouts):
    src = ChaosSource(cutouts, [Noise(0.3, seed=1), CompassError(0.5, seed=2)])
    frames, plain = list(src), list(cutouts)
    assert isinstance(src, FrameSource) and all(isinstance(f, Frame) for f in frames)
    assert len(src) == len(frames) == len(cutouts)
    assert [(f.idx, f.t, f.source, f.scale_prior) for f in frames] == \
           [(f.idx, f.t, f.source, f.scale_prior) for f in plain]
    assert src.fps == cutouts.fps                          # the source's own attributes stay reachable
    for f in frames:
        tr = src.truth(f.idx)                              # truth by idx, untouched: chaos never moves the drone
        assert tr.t == f.t and tr.center == cutouts.truth(f.idx).center
        assert _angle(f.heading_prior_deg, tr.heading_deg) == \
               pytest.approx(abs(CompassError(0.5, seed=2).error_deg(f.idx)), abs=1e-6)
    assert all(f.image.shape == p.image.shape and not np.array_equal(f.image, p.image)
               for f, p in zip(frames, plain, strict=True))


def test_len_counts_the_frames_that_come_out(cutouts):
    src = ChaosSource(cutouts, [Noise(0.2), Drop(0.6, seed=3)])
    n = len(src)
    assert n == len(list(src)) and 0 < n < len(cutouts)
    assert len(src) == n                                   # asking twice gives the same answer


def test_a_source_without_truth_has_none_after_chaos(image):
    src = ChaosSource(_Frames([_frame(image, 0)]), [Noise(0.2)])
    assert len(list(src)) == 1
    with pytest.raises(AttributeError):
        src.truth(0)


def test_chaos_sits_on_a_flight_source(fixture_flight):  # noqa: F811 - the fixture imported above
    d, _ = fixture_flight
    flight = FlightSource(d, fps=0.2, keyframes=True, priors=True)
    comp = CompassError(1.0, seed=5)
    src = ChaosSource(flight, [Gamma(0.5, seed=1), Glare(0.5, seed=2), comp])
    frames, plain = list(src), list(flight)
    assert len(src) == len(frames) == 3 and [f.idx for f in frames] == [6150, 6300, 6450]
    for f, p in zip(frames, plain, strict=True):
        assert f.t == p.t == src.truth(f.idx).t and f.scale_prior == p.scale_prior
        assert f.image.shape == p.image.shape and not np.array_equal(f.image, p.image)
        assert _angle(f.heading_prior_deg, src.truth(f.idx).heading_deg) == \
               pytest.approx(abs(comp.error_deg(f.idx)), abs=1e-6)


# --- 7. bad settings -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [*IMAGE, *STREAM])
@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf"), "0.5", None, True])
def test_a_strength_outside_zero_to_one_is_refused(cls, bad):
    with pytest.raises(ValueError, match="strength"):
        cls(bad)


@pytest.mark.parametrize("cls", [*IMAGE, *STREAM])
@pytest.mark.parametrize("bad", [-1, 1.5, "7", None, True])
def test_a_seed_that_is_not_a_whole_number_is_refused(cls, bad):
    with pytest.raises(ValueError, match="seed"):
        cls(0.5, seed=bad)


@pytest.mark.parametrize("kw", [{"start_s": -1.0}, {"max_s": 0.0}, {"max_s": float("nan")}])
def test_a_jam_needs_a_real_window(kw):
    with pytest.raises(ValueError, match="start_s|max_s"):
        Jam(0.5, **kw)


def test_an_image_degradation_needs_a_bgr_uint8_image():
    with pytest.raises(ValueError, match="uint8"):
        Noise(0.5).image(np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        Blur(0.5).image(np.zeros((8, 8, 3), np.float32))


# --- 8. the check sheets ---------------------------------------------------------------------------------------

def _sheets(*args) -> subprocess.CompletedProcess:
    """Run scripts/chaos_sheet.py as a user would; the caller checks the exit code."""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "chaos_sheet.py"), *map(str, args)],
                          capture_output=True, text=True, check=False, cwd=ROOT)


def _check_sheets(out: Path) -> dict[str, np.ndarray]:
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.png" for n in NAMES)
    sheets = {}
    for n in NAMES:
        img = cv2.imread(str(out / f"{n}.png"))
        assert img is not None, n
        h, w = img.shape[:2]
        # five strengths side by side (or a strip, or a plot): wide enough to read, small enough to open anywhere
        assert 600 <= w <= 4000 and 150 <= h <= 4000, (n, w, h)
        assert img.std() > 10, n                           # something is drawn on it
        sheets[n] = img
    return sheets


def test_sheets_from_the_cutout_generator_need_no_data(tmp_path):
    r = _sheets("--out", tmp_path / "a", "--seed", 5)
    assert r.returncode == 0, r.stderr
    a = _check_sheets(tmp_path / "a")
    assert "seed 5" in r.stdout and str(tmp_path / "a") in r.stdout
    # the seed is on the sheet and decides it: the same seed draws the same sheets, another seed other ones
    assert _sheets("--out", tmp_path / "b", "--seed", 5).returncode == 0
    assert _sheets("--out", tmp_path / "c", "--seed", 6).returncode == 0
    b, c = _check_sheets(tmp_path / "b"), _check_sheets(tmp_path / "c")
    assert all(np.array_equal(a[n], b[n]) for n in NAMES)
    assert all(a[n].shape != c[n].shape or not np.array_equal(a[n], c[n]) for n in NAMES)


def test_sheets_from_a_flight_folder(fixture_flight, tmp_path):  # noqa: F811 - the fixture imported above
    d, _ = fixture_flight
    r = _sheets("--flight", d, "--keyframes", "--fps", 0.2, "--out", tmp_path / "sheets")
    assert r.returncode == 0, r.stderr
    _check_sheets(tmp_path / "sheets")


def test_the_sheet_script_names_a_missing_flight(tmp_path):
    r = _sheets("--flight", tmp_path / "nowhere", "--out", tmp_path / "sheets")
    assert r.returncode == 2 and "nowhere" in r.stderr
    assert not (tmp_path / "sheets").exists() or not list((tmp_path / "sheets").iterdir())
