"""M0-04 · Fake run generator: a scripted flight that renderers and the UI can be built on with no engine.

The "Done when" items from the issue, each tested here:
1. the CLI writes run.json, a run.jsonl that read_run accepts, and one frame per record at its `source`;
2. states go TRACKING → LOST → RELOCALIZING → TRACKING exactly once, with the scripted fixes;
3. every fix is geometrically self-consistent, footprints stay inside the map, the route is smooth with the
   heading along it;
4. frames are the map seen through the true homography (built here independently from the conventions in
   dronenav.types, so a convention mix-up in the generator fails the test);
5. inliers vary, σ grows while LOST and drops after relocalization;
6. same seed → byte-identical output, different seed → different route.

A synthetic textured map keeps the tests fast and independent of data/.
"""
import hashlib
import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.fake_run import FakeRunConfig, make_fake_run
from dronenav.telemetry import read_run, read_run_info
from dronenav.types import TrackMode

ROOT = Path(__file__).resolve().parents[1]
MAP_W, MAP_H = 1400, 900


def _cli(*args) -> subprocess.CompletedProcess:
    """Run scripts/fake_run.py as a user would; the caller checks the exit code."""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "fake_run.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def _texture_map(path: Path, seed: int = 1) -> Path:
    """Smooth random texture: structure at a few-pixel scale, so bilinear sampling is well defined."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 255, (MAP_H, MAP_W, 3)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One default run made through the CLI (also the test of item 1's command line)."""
    d = tmp_path_factory.mktemp("fake")
    map_path = _texture_map(d / "map.png")
    out = d / "run"
    res = _cli("--map", map_path, "--out", out)
    assert res.returncode == 0, res.stderr
    return {"dir": out, "map": map_path, "records": read_run(out / "run.jsonl"), "info": read_run_info(out)}


def _route_shape(run_dir: Path):
    """(largest / smallest step, largest turn per frame in degrees) of the true route."""
    recs = read_run(run_dir / "run.jsonl")
    pts = np.array([[r.extra["truth"]["x"], r.extra["truth"]["y"]] for r in recs])
    heads = np.array([r.extra["truth"]["heading_deg"] for r in recs])
    steps = np.hypot(*np.diff(pts, axis=0).T)
    return steps.max() / steps.min(), np.abs((np.diff(heads) + 180) % 360 - 180).max()


def _lost_drift(run_dir: Path) -> float:
    """How much further from the truth the dead-reckoned position is at the end of the jam than at its start."""
    lost = [r for r in read_run(run_dir / "run.jsonl") if r.track.state is TrackMode.LOST]
    err = [math.hypot(r.track.x - r.extra["truth"]["x"], r.track.y - r.extra["truth"]["y"]) for r in lost]
    return err[-1] - err[0]


def _states(records):
    """The state sequence with repeats collapsed: [TRACKING, LOST, ...]."""
    seq = []
    for r in records:
        if not seq or seq[-1] is not r.track.state:
            seq.append(r.track.state)
    return seq


def _sigma(rec):
    return math.sqrt(max(np.linalg.eigvalsh(rec.track.cov)))


def _true_H(truth, frame_size):
    """Frame px -> map px from the conventions in dronenav.types, independently of the generator.

    Heading is clockwise from map up with y pointing down, so the frame's up vector (0, -1) must land on
    (sin h, -cos h): R = [[cos, -sin], [sin, cos]]. The frame centre (w/2, h/2) lands on the truth position.
    """
    w, h = frame_size
    a = math.radians(truth["heading_deg"])
    sR = truth["scale"] * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([truth["x"], truth["y"]]) - sR @ np.array([w / 2, h / 2])
    return np.array([[*sR[0], t[0]], [*sR[1], t[1]], [0, 0, 1.0]])


def _apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


# --- 1. the run folder -----------------------------------------------------------------------------------------

def test_cli_writes_a_complete_run_folder(run):
    info, recs = run["info"], run["records"]
    assert info.map_path.samefile(run["map"])
    assert info.map_size == (MAP_W, MAP_H) and info.frame_size == (640, 640)
    assert info.generator == "fake_run" and info.seed == 0 and info.fps == 1.0
    assert [r.idx for r in recs] == list(range(50))
    assert all(r.t == pytest.approx(r.idx / info.fps) for r in recs)
    for r in recs:
        img = cv2.imread(str(run["dir"] / r.source))
        assert img is not None, r.source
        assert img.shape == (640, 640, 3)
    assert sorted(p.name for p in (run["dir"] / "frames").iterdir()) == [Path(r.source).name for r in recs]


def test_options_change_the_run(tmp_path):
    m = _texture_map(tmp_path / "map.png")
    cfg = FakeRunConfig(frames=30, fps=5.0, seed=3, frame_size=(320, 240), scale=0.3, jam_start=10, jam_len=3)
    info = make_fake_run(m, tmp_path / "run", cfg)
    recs = read_run(tmp_path / "run" / "run.jsonl")
    assert len(recs) == 30 and info.frame_size == (320, 240) and info.fps == 5.0 and info.seed == 3
    assert [r.idx for r in recs if r.track.state is TrackMode.LOST] == [10, 11, 12]
    assert cv2.imread(str(tmp_path / "run" / recs[0].source)).shape == (240, 320, 3)


@pytest.mark.parametrize("args, words", [
    (["--frames", "8", "--jam-start", "5", "--jam-len", "3"], ["jam"]),    # no room to relocalize and track again
    (["--jam-start", "0"], ["jam"]),                                         # must start after a tracked frame
    (["--scale", "5"], ["map", "small"]),                                    # footprint larger than the map
])
def test_cli_rejects_impossible_settings(tmp_path, args, words):
    m = _texture_map(tmp_path / "map.png")
    res = _cli("--map", m, "--out", tmp_path / "run", *args)
    assert res.returncode != 0
    for w in words:
        assert w in res.stderr.lower(), res.stderr


def test_cli_reports_a_missing_map(tmp_path):
    res = _cli("--map", tmp_path / "nope.png", "--out", tmp_path / "run")
    assert res.returncode != 0 and "nope.png" in res.stderr


# --- 2. the jam script -----------------------------------------------------------------------------------------

def test_states_follow_the_jam_script(run):
    recs = run["records"]
    assert _states(recs) == [TrackMode.TRACKING, TrackMode.LOST, TrackMode.RELOCALIZING, TrackMode.TRACKING]
    lost = [r for r in recs if r.track.state is TrackMode.LOST]
    reloc = [r for r in recs if r.track.state is TrackMode.RELOCALIZING]
    assert len(lost) == 5 and all(r.track.fix is None for r in lost)
    assert len(reloc) == 2
    assert not reloc[0].track.fix.accepted and reloc[0].track.fix.reject_reason
    assert reloc[1].track.fix.accepted
    tracking = [r for r in recs if r.track.state is TrackMode.TRACKING]
    assert all(r.track.fix is not None and r.track.fix.accepted for r in tracking)
    # what the engine did, for the HUD: windowed search while tracking, global search to recover
    assert {r.extra["search"] for r in tracking} == {"window"}
    assert {r.extra["search"] for r in lost} == {"none"}
    assert {r.extra["search"] for r in reloc} == {"global"}


def test_every_state_change_has_an_event_caption(run):
    recs = run["records"]
    for prev, r in itertools.pairwise(recs):
        if r.track.state is not prev.track.state:
            assert isinstance(r.extra.get("event"), str) and r.extra["event"], r.idx
        else:
            assert "event" not in r.extra, r.idx


def test_records_carry_truth_and_matcher_info(run):
    for r in run["records"]:
        assert set(r.extra["truth"]) == {"x", "y", "heading_deg", "scale"}
        assert r.backend == "fake" and r.ms is not None and r.ms > 0
        assert r.interpolated is False


# --- 3. geometry -----------------------------------------------------------------------------------------------

def test_every_fix_is_geometrically_self_consistent(run):
    w, h = run["info"].frame_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)   # TL, TR, BR, BL, as in Fix.footprint
    for r in run["records"]:
        f = r.track.fix
        if f is None:
            continue
        c = np.array([w / 2, h / 2])
        assert np.allclose(_apply(f.H, c[None]), [[f.x, f.y]], atol=1e-6)
        assert np.allclose(_apply(f.H, corners), f.footprint, atol=1e-6)
        up = _apply(f.H, (c + [0, -1])[None])[0] - [f.x, f.y]      # where the frame's up points on the map
        heading_of_H = math.degrees(math.atan2(up[0], -up[1]))
        assert abs((heading_of_H - f.heading_deg + 180) % 360 - 180) < 1e-6   # compared across the 0/360 seam
        assert np.hypot(*up) == pytest.approx(f.scale, rel=1e-9)


def test_fixes_are_near_the_truth(run):
    # accepted fixes are noisy measurements of the true pose, not something unrelated
    for r in run["records"]:
        f, tr = r.track.fix, r.extra["truth"]
        if f is not None and f.accepted:
            assert math.hypot(f.x - tr["x"], f.y - tr["y"]) < 10
            assert abs((f.heading_deg - tr["heading_deg"] + 180) % 360 - 180) < 5


def test_footprints_stay_inside_the_map(run):
    fs = run["info"].frame_size
    for r in run["records"]:
        boxes = [_apply(_true_H(r.extra["truth"], fs), np.array([[0, 0], [fs[0], 0], [fs[0], fs[1]], [0, fs[1]]]))]
        if r.track.fix is not None:
            boxes.append(r.track.fix.footprint)
        for b in boxes:
            assert (b[:, 0] >= 0).all() and (b[:, 0] <= MAP_W).all(), r.idx
            assert (b[:, 1] >= 0).all() and (b[:, 1] <= MAP_H).all(), r.idx


def test_route_is_smooth_and_heading_follows_it(run):
    pts = np.array([[r.extra["truth"]["x"], r.extra["truth"]["y"]] for r in run["records"]])
    heads = np.array([r.extra["truth"]["heading_deg"] for r in run["records"]])
    steps = np.hypot(*np.diff(pts, axis=0).T)
    assert steps.min() > 0 and steps.max() / steps.min() < 3          # no stops, no jumps
    turn = np.abs((np.diff(heads) + 180) % 360 - 180)
    assert turn.max() < 30                                             # degrees per frame
    for i in range(1, len(pts) - 1):                                   # heading along the central difference
        d = pts[i + 1] - pts[i - 1]
        along = math.degrees(math.atan2(d[0], -d[1])) % 360
        assert abs((heads[i] - along + 180) % 360 - 180) < 10, i


def test_velocity_matches_the_route_while_tracking(run):
    recs, fps = run["records"], run["info"].fps
    for prev, r, nxt in zip(recs, recs[1:], recs[2:]):
        if r.track.state is TrackMode.TRACKING:
            v_true = (np.array([nxt.extra["truth"]["x"], nxt.extra["truth"]["y"]]) -
                      np.array([prev.extra["truth"]["x"], prev.extra["truth"]["y"]])) * fps / 2
            assert np.hypot(r.track.vx - v_true[0], r.track.vy - v_true[1]) < 0.25 * np.hypot(*v_true) + 2


# --- 4. frames are the map seen through the true H -------------------------------------------------------------

def _expected_view(map_img, truth, frame_size):
    """Sample the map at the map position of every frame pixel centre (pixel (i, j) is at (i + .5, j + .5))."""
    w, h = frame_size
    jj, ii = np.mgrid[0:h, 0:w]
    q = _apply(_true_H(truth, frame_size), np.c_[ii.ravel() + 0.5, jj.ravel() + 0.5])
    mx = (q[:, 0] - 0.5).reshape(h, w).astype(np.float32)               # map pixel centres back to array index
    my = (q[:, 1] - 0.5).reshape(h, w).astype(np.float32)
    return cv2.remap(map_img, mx, my, cv2.INTER_LINEAR)


def test_frames_are_the_map_seen_through_the_true_homography(run):
    map_img = cv2.imread(str(run["map"]))
    seen = [r for r in run["records"] if r.track.state is not TrackMode.LOST]
    for r in seen[::6]:
        frame = cv2.imread(str(run["dir"] / r.source)).astype(np.float32)
        expected = _expected_view(map_img, r.extra["truth"], run["info"].frame_size).astype(np.float32)
        assert np.abs(frame - expected).mean() < 3.0, r.idx           # grey levels; PNG is lossless


def test_jammed_frames_show_no_map(run):
    map_img = cv2.imread(str(run["map"]))
    for r in (r for r in run["records"] if r.track.state is TrackMode.LOST):
        frame = cv2.imread(str(run["dir"] / r.source), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        expected = cv2.cvtColor(_expected_view(map_img, r.extra["truth"], run["info"].frame_size),
                                cv2.COLOR_BGR2GRAY).astype(np.float32)
        corr = np.corrcoef(frame.ravel(), expected.ravel())[0, 1]
        assert abs(corr) < 0.2, (r.idx, corr)


# --- 5. inliers and σ ------------------------------------------------------------------------------------------

def test_inliers_vary_within_the_task_range(run):
    inl = [r.track.fix.inliers for r in run["records"] if r.track.state is TrackMode.TRACKING]
    assert min(inl) >= 120 and max(inl) <= 384      # the range the 50 task crops have (experiments/data_probe)
    assert max(inl) - min(inl) >= 60


def test_sigma_grows_while_lost_and_drops_after_relocalizing(run):
    recs = run["records"]
    lost = [i for i, r in enumerate(recs) if r.track.state is TrackMode.LOST]
    sig = [_sigma(r) for r in recs]
    assert sig[lost[0]] > sig[lost[0] - 1]
    assert all(b > a for a, b in zip(sig[lost[0]:lost[-1]], sig[lost[0] + 1:lost[-1] + 1]))
    back = next(i for i in range(lost[-1], len(recs)) if recs[i].track.state is TrackMode.TRACKING)
    assert sig[back] < sig[lost[-1]] / 2
    tracking = [s for s, r in zip(sig, recs) if r.track.state is TrackMode.TRACKING]
    assert max(tracking) < min(sig[i] for i in lost)


def test_dead_reckoning_drifts_from_the_truth_while_lost(run):
    assert _lost_drift(run["dir"]) > 5     # coasting on a stale velocity leaves the route


def test_the_drift_shows_for_every_seed(tmp_path):
    # the jam can fall on a nearly straight stretch, where coasting straight would not drift at all; the UI
    # and the HUD video need to show the drift whatever the seed (seed 11 drifted only 4 px before the fix; seed
    # 30 still did with a bias that turned the noisy velocity estimate, because the noise ate half the bias)
    m = _texture_map(tmp_path / "map.png")
    for seed in range(40):
        make_fake_run(m, tmp_path / f"s{seed}", FakeRunConfig(seed=seed, frame_size=(96, 96), scale=1.7))
        assert _lost_drift(tmp_path / f"s{seed}") > 5, seed


def test_a_portrait_map_gets_a_smooth_route_too(tmp_path):
    # the route runs along the map's long side; across a tall map it zig-zagged (step ratio 6, 37° per frame)
    m = tmp_path / "tall.png"
    cv2.imwrite(str(m), cv2.rotate(cv2.imread(str(_texture_map(tmp_path / "wide.png"))), cv2.ROTATE_90_CLOCKWISE))
    make_fake_run(m, tmp_path / "run")
    ratio, turn = _route_shape(tmp_path / "run")
    assert ratio < 3 and turn < 30
    fs = read_run_info(tmp_path / "run").frame_size
    for r in read_run(tmp_path / "run" / "run.jsonl"):
        b = _apply(_true_H(r.extra["truth"], fs), np.array([[0, 0], [fs[0], 0], [fs[0], fs[1]], [0, fs[1]]]))
        assert (b >= 0).all() and (b[:, 0] <= MAP_H).all() and (b[:, 1] <= MAP_W).all(), r.idx


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_a_frame_that_cannot_be_written_is_an_error(tmp_path):
    # cv2.imwrite only logs a warning and returns False: the run looked complete with no frames on disk
    m = _texture_map(tmp_path / "map.png")
    frames = tmp_path / "run" / "frames"
    frames.mkdir(parents=True)
    frames.chmod(0o500)
    try:
        with pytest.raises(OSError, match="frame_00000.png"):
            make_fake_run(m, tmp_path / "run", FakeRunConfig(frames=12, jam_start=3, jam_len=2))
    finally:
        frames.chmod(0o700)


# --- 6. determinism --------------------------------------------------------------------------------------------

def _digest(run_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(run_dir).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_same_seed_same_bytes_different_seed_different_route(tmp_path):
    m = _texture_map(tmp_path / "map.png")
    cfg = FakeRunConfig(frames=12, jam_start=4, jam_len=2)
    make_fake_run(m, tmp_path / "a", cfg)
    make_fake_run(m, tmp_path / "b", cfg)
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    make_fake_run(m, tmp_path / "c", FakeRunConfig(frames=12, jam_start=4, jam_len=2, seed=1))
    ta = [r.extra["truth"]["x"] for r in read_run(tmp_path / "a" / "run.jsonl")]
    tc = [r.extra["truth"]["x"] for r in read_run(tmp_path / "c" / "run.jsonl")]
    assert ta != tc


# --- on the real task map, when data/ is present ---------------------------------------------------------------

TASK_MAP = ROOT / "data" / "hackatons-data" / "global_map.png"


@pytest.mark.data
@pytest.mark.skipif(not TASK_MAP.exists(), reason="data/ not downloaded")
def test_default_run_on_the_task_map(tmp_path):
    info = make_fake_run(TASK_MAP, tmp_path / "run")
    recs = read_run(tmp_path / "run" / "run.jsonl")
    assert len(recs) == 50 and info.map_size == (3645, 1845)
    for r in recs:
        if r.track.fix is not None:
            fp = r.track.fix.footprint
            assert (fp >= 0).all() and (fp[:, 0] <= 3645).all() and (fp[:, 1] <= 1845).all()
