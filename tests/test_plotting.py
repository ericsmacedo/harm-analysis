# MIT License
#
# Copyright (c) 2026 ericsmacedo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests for the spectrum plots drawn by harm_analysis and spec_analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Annotation

from harm_analysis import harm_analysis, spec_analysis

N = 4096
T = np.arange(N)


def _texts(ax):
    return [text.get_text() for text in ax.texts]


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


@pytest.fixture
def ax():
    """A fresh matplotlib axes, closed after the test."""
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _signal(noise_std=1e-4):
    x = np.cos(2 * np.pi * 0.1003 * T) + 0.01 * np.cos(2 * np.pi * 0.2006 * T) + 0.005 * np.cos(2 * np.pi * 0.3009 * T)
    return x + np.random.default_rng(0).normal(0, noise_std, N)


def test_harmonics_are_numbered_by_order(ax):
    """The 2nd harmonic is labeled H2, the 3rd H3 (as in MATLAB's thd)."""
    results = harm_analysis(_signal(), n_harm=2, plot=True, ax=ax)

    texts = _texts(ax)
    assert any(t.startswith("F\n") for t in texts)
    assert f"H2\n{results.harm_db[0]:.1f} dB" in texts
    assert f"H3\n{results.harm_db[1]:.1f} dB" in texts
    assert not any(t.startswith("H1\n") for t in texts)


def test_peak_labels_show_the_reported_levels(ax):
    """The fundamental label shows fund_db."""
    results = harm_analysis(_signal(), plot=True, ax=ax)

    assert f"F\n{results.fund_db:.1f} dB" in _texts(ax)


def test_noise_label_is_the_reported_noise_with_bw(ax):
    """With bw, the integrated-noise label shows the in-band noise_db, not the end of the curve."""
    results = harm_analysis(_signal(noise_std=1e-2), n_harm=0, bw=0.2, plot=True, ax=ax)

    assert f"Integrated noise ({results.noise_db:.1f} dB)" in _legend_labels(ax)
    assert "bw" in _legend_labels(ax)


def test_noise_label_without_bw(ax):
    """Without bw, the integrated-noise label is the total noise."""
    results = harm_analysis(_signal(), plot=True, ax=ax)

    assert f"Integrated noise ({results.noise_db:.1f} dB)" in _legend_labels(ax)
    assert "bw" not in _legend_labels(ax)


def test_noise_plus_harmonics_curve_only_with_harmonics(ax):
    """The 'noise + harmonics' curve is drawn only when there are harmonics."""
    harm_analysis(_signal(), n_harm=0, plot=True, ax=ax)

    assert not any(label.startswith("Integrated noise + harmonics") for label in _legend_labels(ax))


def test_spec_plot_labels_tones(ax):
    """Each detected tone is labeled with its index and level."""
    results = spec_analysis(_signal(), plot=True, ax=ax)

    texts = _texts(ax)
    for i, level in enumerate(results.tones_db):
        assert f"T{i}\n{level:.1f} dB" in texts
    assert f"Integrated noise ({results.noise_db:.1f} dB)" in _legend_labels(ax)


@pytest.mark.parametrize("func", [harm_analysis, spec_analysis])
def test_axes_styling(ax, func):
    """Axis labels, headroom above the highest peak, and only the given axes are styled."""
    fig = ax.figure
    facecolor = fig.get_facecolor()

    func(_signal(), plot=True, ax=ax)

    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Power [dB]"
    assert not ax.spines["top"].get_visible()
    assert ax.get_ylim()[1] >= 10 * np.log10(0.5) + 15  # headroom for the peak label
    assert fig.get_facecolor() == facecolor


def test_plot_on_a_dark_axes_background():
    """Plotting works on a dark style; labels get a halo in the axes background color."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots()
        harm_analysis(_signal(), plot=True, ax=ax)
        fig.canvas.draw()  # renders the labels and their halos
        background = ax.get_facecolor()
        labels = [text for text in ax.texts if text.get_path_effects()]
        plt.close(fig)

    assert background[:3] == (0.0, 0.0, 0.0)
    assert labels  # every label has a halo


def _label_boxes(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # ty: ignore[unresolved-attribute]
    return [text.get_window_extent(renderer) for text in ax.texts if "dB" in text.get_text()]


def _audio_tone(fs=48000, n=16384):
    """A 1 kHz tone with five harmonics, close together on a log frequency axis."""
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 1000 * t) + sum(
        10 ** (-(40 + 8 * k) / 20) * np.sin(2 * np.pi * 1000 * (k + 2) * t) for k in range(5)
    )
    return x + np.random.default_rng(0).normal(0, 1e-4, n)


@pytest.mark.parametrize("log_first", [False, True])
def test_peak_labels_do_not_overlap_on_a_log_frequency_axis(log_first):
    """Labels of close harmonics don't overlap, also when the log scale is set after plotting."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if log_first:
        ax.set_xscale("log")
    harm_analysis(_audio_tone(), fs=48000, plot=True, ax=ax)
    ax.set_xscale("log")
    boxes = _label_boxes(fig, ax)
    plt.close(fig)

    assert len(boxes) == 6  # F, H2 ... H6
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not a.overlaps(b)


def test_peak_labels_move_back_down_when_zoomed_in():
    """Labels are placed at each draw: after zooming in, a label with room sits next to its peak."""
    fig, ax = plt.subplots(figsize=(8, 4))
    harm_analysis(_audio_tone(), fs=48000, plot=True, ax=ax)
    ax.set_xscale("log")
    fig.canvas.draw()
    labels = [text for text in ax.texts if isinstance(text, Annotation) and text.get_text().startswith("H")]
    lifted = [label for label in labels if label.xyann[1] > 5]
    assert lifted  # some labels had to move up to avoid their neighbors
    label = lifted[0]
    ax.set_xlim(label.xy[0] * 0.98, label.xy[0] * 1.02)
    fig.canvas.draw()
    offset = label.xyann[1]
    plt.close(fig)

    assert offset == 5
