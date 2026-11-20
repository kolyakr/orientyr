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


def test_names_without_numbers_are_numbered_by_position(tmp_path):
    for name in ("c.png", "a.png", "b.png"):
        _img(tmp_path / name)
    frames = list(FolderSource(tmp_path))
    assert [Path(f.source).name for f in frames] == ["a.png", "b.png", "c.png"]
    assert [f.idx for f in frames] == [0, 1, 2]


def test_repeated_numbers_are_numbered_by_position_in_a_fixed_order(tmp_path):
    # the number is not a frame number when it repeats (take1_x, take1_y); f_1 and f_01 tie on the number, so
    # the name decides, not the filesystem
    for name in ("take1_y.png", "take1_x.png", "f_01.png", "f_1.png"):
        _img(tmp_path / name)
    frames = list(FolderSource(tmp_path))
    assert [Path(f.source).name for f in frames] == ["f_01.png", "f_1.png", "take1_x.png", "take1_y.png"]
    assert [f.idx for f in frames] == [0, 1, 2, 3]


def test_a_stray_file_without_a_number_is_an_error_not_a_renumbering(tmp_path):
    # one preview.png among keyframes used to renumber the whole folder: frame_00030 at 0.033 s instead of 1 s
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30}))
    for n in (0, 30, 60):
        _img(tmp_path / "frames" / f"frame_{n:05d}.png")
    _img(tmp_path / "frames" / "preview.png")
    with pytest.raises(ValueError, match="preview.png"):
        FolderSource(tmp_path / "frames")


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


@pytest.mark.parametrize("meta", ['{"fps": 0}', '{"fps": "fast"}', '{"fps": true}', "not json", "[30]"])
def test_a_meta_json_with_a_broken_fps_is_a_clear_error(tmp_path, meta):
    (tmp_path / "meta.json").write_text(meta)
    _img(tmp_path / "frames" / "frame_00000.png")
    with pytest.raises(ValueError, match="meta.json"):
        FolderSource(tmp_path / "frames")


def test_a_meta_json_that_says_nothing_about_fps_is_ignored(tmp_path):
    # another tool's meta.json (no "fps" key) is not about timing: the next place, or 1 fps, applies
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30}))
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "meta.json").write_text(json.dumps({"frames": 10}))
    _img(tmp_path / "frames" / "frame_00030.png")
    assert [f.t for f in FolderSource(tmp_path / "frames")] == [1.0]


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
    _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))                # creates maps/ first
    side = _sidecar(tmp_path / "maps" / "m.json", (30, 20), hires_size=(60, 40))
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


# --- review ----------------------------------------------------------------------------------------------------

def test_frames_follow_their_numbers_so_time_never_runs_back(tmp_path):
    # the number is the frame's time: ordered by name, a_10, b_2, c_5 ran 10 s, 2 s, 5 s and the telemetry
    # writer (idx strictly increasing) would refuse the run
    for name in ("a_10.png", "b_2.png", "c_5.png"):
        _img(tmp_path / name)
    frames = list(FolderSource(tmp_path))
    assert [f.idx for f in frames] == [2, 5, 10]
    assert [Path(f.source).name for f in frames] == ["b_2.png", "c_5.png", "a_10.png"]
    assert [f.t for f in frames] == sorted(f.t for f in frames)


@pytest.mark.parametrize("size", [5, "30x20", [30], [30, "20"], [30, -20], None, True])
def test_a_malformed_size_in_the_sidecar_is_a_clear_error(tmp_path, size):
    p = _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))
    (tmp_path / "maps" / "m.json").write_text(json.dumps({"map": {"file": "maps/m.png", "size_px": size}}))
    with pytest.raises(ValueError, match=r"m\.json.*size_px"):
        load_map(p)


def test_an_unrelated_broken_json_in_maps_does_not_stop_a_hires_layer(tmp_path):
    _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))
    side = _sidecar(tmp_path / "maps" / "m.json", (30, 20), hires_size=(60, 40))
    (tmp_path / "maps" / "notes.json").write_text("{not json")
    (tmp_path / "maps" / "list.json").write_text("[1, 2]")
    m = load_map(_img(tmp_path / "hires" / "m_z18.png", shape=(40, 60, 3)))
    assert m.sidecar == side and m.level == side["hires"]



def test_odd_digits_in_names_do_not_crash(tmp_path):
    # "²" is a digit to str.isdigit() but not to the \d the names are split on
    for name in ("\u00b21.png", "x_2.png"):
        _img(tmp_path / name)
    assert len(list(FolderSource(tmp_path))) == 2


def test_a_folder_named_like_an_image_is_not_a_frame(tmp_path):
    _img(tmp_path / "f_1.png")
    (tmp_path / "f_9.png").mkdir()
    assert [f.idx for f in FolderSource(tmp_path)] == [1]


def test_upper_case_extensions_are_images(tmp_path):
    for n in (1, 2):
        _img(tmp_path / f"f_{n}.png").rename(tmp_path / f"F_{n}.PNG")
    assert len(FolderSource(tmp_path)) == 2


@pytest.mark.parametrize("fps", [True, float("inf"), float("nan"), "30"])
def test_fps_arguments_that_are_not_real_frame_rates_are_refused(tmp_path, fps):
    _img(tmp_path / "f_0.png")
    with pytest.raises(ValueError, match="fps"):
        FolderSource(tmp_path, fps=fps)


def test_numpy_frame_rates_are_fine(tmp_path):
    _img(tmp_path / "f_30.png")
    assert [f.t for f in FolderSource(tmp_path, fps=np.int64(30))] == [1.0]
    assert [f.t for f in FolderSource(tmp_path, fps=np.float32(15))] == [2.0]


@pytest.mark.parametrize("raw", [b'{"fps": 1' + b"0" * 400 + b"}", b'{"fps": 30, "note": "\xff\xfe"}'.replace(b"\\x", b"\\x"),
                                 b"\xff\xfe not utf-8"])
def test_meta_json_read_errors_name_the_file(tmp_path, raw):
    (tmp_path / "meta.json").write_bytes(raw)
    _img(tmp_path / "frames" / "frame_00000.png")
    with pytest.raises(ValueError, match="meta.json"):
        FolderSource(tmp_path / "frames")


def _exif_rotated_jpeg(path: Path, shape=(20, 40, 3)):
    """A JPEG whose EXIF says "rotate 90°" (orientation 6), the way phones save portrait photos."""
    ok, buf = cv2.imencode(".jpg", np.zeros(shape, np.uint8))
    assert ok
    ifd = b"II*\x00\x08\x00\x00\x00" + b"\x01\x00" + b"\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00" + b"\x00" * 4
    app1 = b"Exif\x00\x00" + ifd
    data = buf.tobytes()
    path.write_bytes(data[:2] + b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1 + data[2:])
    return path


def test_a_map_is_read_as_stored_not_rotated_by_exif(tmp_path):
    # map pixels are coordinates: an EXIF rotation would turn the map and break its sidecar's size
    m = load_map(_exif_rotated_jpeg(tmp_path / "map.jpg"))
    assert m.image.shape == (20, 40, 3)


def test_a_hires_layer_takes_the_sidecar_that_names_it_not_the_first_one(tmp_path):
    # irpin_2017.json sorts first; a hires layer must get its own year's georeference, never another's
    _img(tmp_path / "maps" / "a2017.png", shape=(20, 30, 3))
    _sidecar(tmp_path / "maps" / "a2017.json", (30, 20), hires_size=(60, 40), hires_file="hires/a2017_z18.png")
    _img(tmp_path / "maps" / "b2025.png", shape=(20, 30, 3))
    side = _sidecar(tmp_path / "maps" / "b2025.json", (30, 20), hires_size=(60, 40), hires_file="hires/b2025_z18.png")
    m = load_map(_img(tmp_path / "hires" / "b2025_z18.png", shape=(40, 60, 3)))
    assert m.sidecar == side and m.level["file"] == "hires/b2025_z18.png"


def test_two_sidecars_claiming_one_layer_is_an_error(tmp_path):
    for name in ("a", "b"):
        _img(tmp_path / "maps" / f"{name}.png", shape=(20, 30, 3))
        _sidecar(tmp_path / "maps" / f"{name}.json", (30, 20), hires_size=(60, 40), hires_file="hires/x_z18.png")
    with pytest.raises(ValueError, match=r"a\.json.*b\.json|b\.json.*a\.json"):
        load_map(_img(tmp_path / "hires" / "x_z18.png", shape=(40, 60, 3)))


def test_the_block_that_names_the_file_wins_over_another_of_the_same_size(tmp_path):
    p = _img(tmp_path / "maps" / "m.png", shape=(20, 30, 3))
    (tmp_path / "maps" / "m.json").write_text(json.dumps({
        "camera": {"size_px": [30, 20]},                                 # same size, but not this file
        "map": {"file": "maps/m.png", "zoom": 17, "size_px": [30, 20]}}))
    assert load_map(p).level["zoom"] == 17


@pytest.mark.data
@pytest.mark.skipif(not (IRPIN / "hires" / "irpin_2025_z18.png").exists(), reason="data/ not downloaded")
def test_the_irpin_hires_layer_gets_its_own_years_zoom_18_block():
    m = load_map(IRPIN / "hires" / "irpin_2025_z18.png")
    assert m.level["zoom"] == 18 and m.level["file"] == "hires/irpin_2025_z18.png"
    assert m.sidecar["capture"]["date"].startswith("2025")
