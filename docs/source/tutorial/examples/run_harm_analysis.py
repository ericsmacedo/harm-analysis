'''Example usage of the harm_analysis function'''
import numpy as np
import matplotlib.pyplot as plt
from harm_analysis import harm_analysis


# test signal
N = 4096
FS = 1000
t = np.arange(0, N/FS, 1/FS)
F1 = 100.13

noise = np.random.normal(loc=0, scale=10**(-70/20), size=len(t))

# Test signal
# Tone with harmonics, DC and white gaussian noise
x = noise + 0.1234 + 2*np.cos(2*np.pi*F1*t) + 0.01*np.cos(2*np.pi*F1*2*t) +\
    0.005*np.cos(2*np.pi*F1*3*t)

# Use the harm_analysis function
fig, ax = plt.subplots()
results, ax = harm_analysis(x, FS=FS, plot=True, ax=ax)

print("Function results:")
for key, value in results.items():
    print(f"{key.ljust(10)} [dB]: {value}")

# Show plot
ax.set_title('Harmonic analysis example')
plt.show()
