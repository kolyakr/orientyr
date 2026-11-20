"""Simulation (M6): pre-rendered flights played with ground truth, a cut-out generator for tests, and the chaos
layer that degrades either at run time; and a simulated GPS receiver, honest or spoofed (M5-07)."""
from dronenav.sim.chaos import (
    DEGRADATIONS,
    Blur,
    ChaosSource,
    CompassError,
    Degradation,
    Drop,
    Gamma,
    Glare,
    ImageDegradation,
    Jam,
    Jpeg,
    Noise,
    Occlusion,
    degrade,
)
from dronenav.sim.cutout import CutoutPose, CutoutSource, cutout_homography, cutout_route, render_cutout
from dronenav.sim.flight import FlightSource
from dronenav.sim.gps import GpsSample, TruthPoint, WalkOff, read_truth_track, simulate_gps
from dronenav.sim.truth import FlightTruth, centre_scale

__all__ = ["DEGRADATIONS", "Blur", "ChaosSource", "CompassError", "CutoutPose", "CutoutSource", "Degradation",
           "Drop", "FlightSource", "FlightTruth", "Gamma", "Glare", "GpsSample", "ImageDegradation", "Jam", "Jpeg",
           "Noise", "Occlusion", "TruthPoint", "WalkOff", "centre_scale", "cutout_homography", "cutout_route",
           "degrade", "read_truth_track", "render_cutout", "simulate_gps"]
