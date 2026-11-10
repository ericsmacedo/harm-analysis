# MIT License
#
# Copyright (c) 2025-2026 ericsmacedo
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
"""Result types returned by the analysis functions."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HarmResult:
    """Results of :func:`harm_analysis`.

    All powers are in dB relative to the input signal's units squared, and all
    frequencies are in the same units as ``fs``. A power of zero is reported as
    ``-inf`` dB.

    Attributes
    ----------
    fund_db : float
        Fundamental power in decibels.
    fund_freq : float
        Frequency of the fundamental tone.
    harm_freq : numpy.ndarray
        Frequency of each harmonic. Empty if no harmonics were analyzed.
    harm_db : numpy.ndarray
        Power of each harmonic in decibels. Empty if no harmonics were analyzed.
    dc_db : float
        DC power in decibels.
    noise_db : float
        Noise power in decibels (excluding DC, the fundamental and the harmonics).
    thd_db : float
        Total harmonic distortion in decibels. ``numpy.nan`` if ``n_harm`` is 0 or if
        all harmonics are outside the bandwidth.
    snr_db : float
        Signal-to-noise ratio in decibels.
    sinad_db : float
        Signal-to-noise-and-distortion ratio in decibels.
    thdn_db : float
        Total harmonic distortion plus noise in decibels.
    total_noise_db : float
        Noise power in decibels integrated over the whole spectrum, from DC to fs/2,
        also beyond `bw` (DC, the fundamental and the analyzed harmonics excluded;
        harmonics above `bw` are not analyzed, so they count as noise here). Equal to
        `noise_db` when `bw` is not given.
    """  # noqa: D416

    fund_db: float
    fund_freq: float
    harm_freq: NDArray[np.float64]
    harm_db: NDArray[np.float64]
    dc_db: float
    noise_db: float
    thd_db: float
    snr_db: float
    sinad_db: float
    thdn_db: float
    total_noise_db: float

    def to_dict(self) -> dict[str, Any]:
        """Return the results as a dictionary of ``{field name: value}``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class SpecResult:
    """Results of :func:`spec_analysis`.

    All powers are in dB relative to the input signal's units squared, and all
    frequencies are in the same units as ``fs``. A power of zero is reported as
    ``-inf`` dB.

    Attributes
    ----------
    dc : float
        DC level, in the same units as the input signal.
    dc_db : float
        DC power in decibels.
    noise_db : float
        Noise power in decibels (excluding DC and the detected tones).
    tones_db : numpy.ndarray
        Power of each detected tone in decibels. Empty if no tones were detected.
    tones_freq : numpy.ndarray
        Frequency of each detected tone. Empty if no tones were detected.
    """  # noqa: D416

    dc: float
    dc_db: float
    noise_db: float
    tones_db: NDArray[np.float64]
    tones_freq: NDArray[np.float64]

    def to_dict(self) -> dict[str, Any]:
        """Return the results as a dictionary of ``{field name: value}``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
