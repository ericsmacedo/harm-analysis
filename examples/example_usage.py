"""Example usage of the harm_analysis function."""

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis import harm_analysis

# test signal
N = 4096
fs = 1000
t = np.arange(0, N / fs, 1 / fs)
F1 = 100.13

noise = np.random.normal(loc=0, scale=10 ** (-70 / 20), size=len(t))

# Test signal
# Tone with harmonics, DC and white gaussian noise
x = (
    noise
    + 0.1234
    + 2 * np.cos(2 * np.pi * F1 * t) # Fundamental
    + 0.01 * np.cos(2 * np.pi * F1 * 2 * t)
    + 0.005 * np.cos(2 * np.pi * F1 * 3 * t)
)

# Use the harm_analysis function
fig, ax = plt.subplots()
parameters, ax = harm_analysis(x, fs=fs, plot=True, ax=ax)

print("Parameters:")
for key, value in parameters.items():
    print(f"{key}: {value}")

# Show plot
ax.set_title("Harmonic analysis example")
plt.show()
