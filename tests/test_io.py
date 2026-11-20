"""M1-01 · Folder frame source and map loader.

"Done when: iterating data/hackatons-data/crops/ yields 50 frames in flight order (crop_0 … crop_49), and iterating
an Irpin frames/ folder (frame_00000.png, …) yields the keyframes with t = index / fps from meta.json." The rules
behind it (issue spec update) are tested on small synthetic folders; the real data is checked when present.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.io import FolderSource, MapImage, load_map
from dronenav.types import Frame, FrameSource

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data" / "hackatons-data" / "crops"
TASK_MAP = ROOT / "data" / "hackatons-data" / "global_map.png"
IRPIN = ROOT / "data" / "irpin"


def _img(path: Path, value: int = 0, shape=(12, 16, 3)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full(shape, value, np.uint8))
    return path


# --- FolderSource: order and numbering ------------------------------------------------------------------------

def test_natural_order_and_frame_numbers(tmp_path):
    for n in (10, 2, 0, 1, 23):
        _img(tmp_path / f"crop_{n}.png", value=n)
    frames = list(FolderSource(tmp_path))
    assert [f.idx for f in frames] == [0, 1, 2, 10, 23]                 # crop_2 before crop_10
    assert [f.t for f in frames] == [0.0, 1.0, 2.0, 10.0, 23.0]          # no meta.json: 1 fps
    assert [int(f.image[0, 0, 0]) for f in frames] == [0, 1, 2, 10, 23]  # each frame is its own file
    assert [Path(f.source).name for f in frames] == ["crop_0.png", "crop_1.png", "crop_2.png", "crop_10.png",
                                                     "crop_23.png"]


def test_names_without_unique_numbers_are_numbered_by_position(tmp_path):
    for name in ("b.png", "a.png", "take1_x.png", "take1_y.png"):     # no number, or the same number twice
        _img(tmp_path / name)
    frames = list(FolderSource(tmp_path))
    assert [Path(f.source).name for f in frames] == ["a.png", "b.png", "take1_x.png", "take1_y.png"]
    assert [f.idx for f in frames] == [0, 1, 2, 3]


def test_the_last_number_in_a_name_is_the_frame_number(tmp_path):
    for n in (5, 40, 120):
        _img(tmp_path / f"flight2_cam1_{n:05d}.png")
    assert [f.idx for f in FolderSource(tmp_path)] == [5, 40, 120]


def test_common_image_formats_any_case_and_nothing_else(tmp_path):
    for name in ("f_1.png", "f_2.JPG", "f_3.jpeg", "f_4.bmp", "f_5.tif", "f_6.TIFF", "f_7.webp"):
        _img(tmp_path / name)
    (tmp_path / ".DS_Store").write_bytes(b"\0\1")
    (tmp_path / "notes.txt").write_text("not a frame")
    (tmp_path / "meta.json").write_text("{}")
    _img(tmp_path / ".hidden_9.png")
    (tmp_path / "sub").mkdir()
    _img(tmp_path / "sub" / "f_8.png")                                   # not recursive
    assert [f.idx for f in FolderSource(tmp_path)] == [1, 2, 3, 4, 5, 6, 7]


def test_frames_are_bgr_uint8_whatever_the_file(tmp_path):
    _img(tmp_path / "f_0.png", shape=(12, 16))                            # grey
    _img(tmp_path / "f_1.png", shape=(12, 16, 4))                         # with alpha
    for f in FolderSource(tmp_path):
        assert isinstance(f, Frame) and f.image.dtype == np.uint8 and f.image.shape == (12, 16, 3)


def test_it_is_a_frame_source_with_a_length(tmp_path):
    for n in range(3):
        _img(tmp_path / f"f_{n}.png")
    src = FolderSource(tmp_path)
    assert isinstance(src, FrameSource) and len(src) == 3
    assert [f.idx for f in src] == [f.idx for f in src]                   # can be iterated again


# --- FolderSource: time ----------------------------------------------------------------------------------------

def test_fps_comes_from_meta_json_in_the_folder_or_its_parent(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30, "frames": 13260}))
    for n in (0, 30, 60, 90):
        _img(tmp_path / "frames" / f"frame_{n:05d}.png")
    frames = list(FolderSource(tmp_path / "frames"))
    assert [f.t for f in frames] == [0.0, 1.0, 2.0, 3.0]                # t = index / fps
    (tmp_path / "frames" / "meta.json").write_text(json.dumps({"fps": 10}))
    assert [f.t for f in FolderSource(tmp_path / "frames")] == [0.0, 3.0, 6.0, 9.0]   # the folder's own wins


def test_the_fps_argument_overrides_meta_json(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30}))
    _img(tmp_path / "frames" / "frame_00030.png")
    assert [f.t for f in FolderSource(tmp_path / "frames", fps=15)] == [2.0]


@pytest.mark.parametrize("meta", ['{"fps": 0}', '{"fps": "fast"}', "not json", '{"frames": 10}'])
def test_a_meta_json_without_a_usable_fps_is_a_clear_error(tmp_path, meta):
    (tmp_path / "meta.json").write_text(meta)
    _img(tmp_path / "frames" / "frame_00000.png")
    with pytest.raises(ValueError, match="meta.json"):
        FolderSource(tmp_path / "frames")


@pytest.mark.parametrize("fps", [0, -1])
def test_a_bad_fps_argument_is_refused(tmp_path, fps):
    _img(tmp_path / "f_0.png")
    with pytest.raises(ValueError, match="fps"):
        FolderSource(tmp_path, fps=fps)


# --- FolderSource: errors --------------------------------------------------------------------------------------

def test_a_missing_folder_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        FolderSource(tmp_path / "nope")


def test_a_folder_without_images_is_a_clear_error(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="no images"):
        FolderSource(tmp_path)


def test_an_unreadable_image_is_named(tmp_path):
    _img(tmp_path / "f_0.png")
    (tmp_path / "f_1.png").write_bytes(b"not a png")
    frames = iter(FolderSource(tmp_path))
    next(frames)                                                          # lazily: the good one still comes first
    with pytest.raises(ValueError, match="f_1.png"):
        next(frames)


# --- load_map --------------------------------------------------------------------------------------------------

def _sidecar(path: Path, size, hires_size=None, hires_file="hires/m_z18.png"):
    d = {"capture": {"date": "2025-04-11"}, "map": {"file": "maps/m.png", "zoom": 17, "size_px": list(size)}}
    if hires_size:
        d["hires"] = {"file": hires_file, "zoom": 18, "size_px": list(hires_size)}
    path.write_text(json.dumps(d))
    return d


def test_a_map_without_a_sidecar_has_no_georeference(tmp_path):
    m = load_map(_img(tmp_path / "global_map.png", shape=(20, 30, 3)))
    assert isinstance(m, MapImage) and m.image.shape == (20, 30, 3) and m.image.dtype == np.uint8
    assert m.sidecar is None and m.level is None and m.path.name == "global_map.png"


def test_the_sidecar_next_to_the_map_and_its_matching_block(tmp_path):
    p = _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))
    side = _sidecar(tmp_path / "maps" / "m.json", (30, 20), hires_size=(60, 40))
    m = load_map(p)
    assert m.sidecar == side and m.level == side["map"]                  # size_px is [w, h]


def test_a_hires_layer_finds_its_block_in_the_dataset_maps_folder(tmp_path):
    side = _sidecar(tmp_path / "maps" / "m.json", (30, 20), hires_size=(60, 40))
    _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))
    m = load_map(_img(tmp_path / "hires" / "m_z18.png", shape=(40, 60, 3)))
    assert m.sidecar == side and m.level == side["hires"]


def test_a_sidecar_that_does_not_match_the_image_is_refused(tmp_path):
    p = _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))
    _sidecar(tmp_path / "maps" / "m.json", (300, 200))
    with pytest.raises(ValueError, match=r"30 x 20.*m\.json"):
        load_map(p)


def test_a_grey_map_is_loaded_as_bgr(tmp_path):
    assert load_map(_img(tmp_path / "g.png", shape=(20, 30))).image.shape == (20, 30, 3)


def test_a_missing_or_unreadable_map_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.png"):
        load_map(tmp_path / "nope.png")
    (tmp_path / "bad.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="bad.png"):
        load_map(tmp_path / "bad.png")


# --- the real data, when present -------------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.skipif(not CROPS.exists(), reason="data/ not downloaded")
def test_the_task_crops_come_in_flight_order():
    frames = list(FolderSource(CROPS))
    assert len(frames) == 50
    assert [Path(f.source).name for f in frames] == [f"crop_{i}.png" for i in range(50)]
    assert [f.idx for f in frames] == list(range(50))
    assert all(f.image.shape == (640, 640, 3) for f in frames)


def _complete_flight():
    """A flight folder whose keyframes are all there (the local download may be partial)."""
    for d in sorted(IRPIN.glob("flights/*/*/frames")):
        n = len(list(d.glob("frame_*.png")))
        if n > 100:
            return d
    return None


@pytest.mark.data
@pytest.mark.skipif(_complete_flight() is None, reason="no complete Irpin flight in data/")
def test_irpin_keyframes_carry_the_times_of_their_truth_records():
    folder = _complete_flight()
    fps = json.loads((folder.parent / "meta.json").read_text())["fps"]
    truth = {r["i"]: r["t"] for r in map(json.loads, (folder.parent / "truth.jsonl").read_text().splitlines())}
    frames = list(FolderSource(folder))
    assert [f.idx for f in frames] == sorted(f.idx for f in frames)
    for f in frames:
        assert f.t == pytest.approx(f.idx / fps)
        assert f.t == pytest.approx(truth[f.idx], abs=1e-6), f.source    # the flight's own clock agrees


@pytest.mark.data
@pytest.mark.skipif(not TASK_MAP.exists(), reason="data/ not downloaded")
def test_the_task_map_loads_without_a_georeference():
    m = load_map(TASK_MAP)
    assert m.image.shape == (1845, 3645, 3) and m.sidecar is None and m.level is None


@pytest.mark.data
@pytest.mark.skipif(not (IRPIN / "maps" / "irpin_2025.png").exists(), reason="data/ not downloaded")
def test_the_irpin_map_finds_its_zoom_17_block():
    m = load_map(IRPIN / "maps" / "irpin_2025.png")
    assert m.level["zoom"] == 17 and m.level["size_px"] == [3936, 3936] and m.sidecar["capture"]["sensor"]
