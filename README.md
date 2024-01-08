# Introduction 
The harmonic analysis function uses an FFT to estimate the following parameters from a signal containing a tone:
- THD and THD+N
- Fundamental power and frequency location
- Noise power
- SNR, SINAD
- DC level
- Total integrated noise (everything except DC and the fundamental)

# Installation

Download the git repository:
```
git clone https://github.com/ericsmacedo/harm_analysis.git
```
Go inside the git project root directory and run:
```
pip install .
```
After installing the package, the function should be available via import:
```python
from harm_analysis import harm_analysis
```
# Example usage

```python
'''Example usage of the harm_analysis function'''
import numpy as np
import matplotlib.pyplot as plt
import pprint
from harm_analysis import harm_analysis

# test signal
N = 2048
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
```

The code above outputs:
```
Function results:
fund_db    [dB]: 3.0103153618915335
fund_freq  [dB]: 100.1300002671261
dc_db      [dB]: -18.174340815733466
noise_db   [dB]: -69.86388900477726
thd_db     [dB]: -45.0412474024929
snr_db     [dB]: 72.87420436666879
sinad_db   [dB]: 45.034100280257974
thdn_db    [dB]: -45.034100280257974
total_noise_and_dist [dB]: -42.023784918366445
```

![Harmonic Analysis plot](docs/harm_analysis_out_example.png)

## Command line interface 
Installing the package also installs a command line interface, that allows the user to run the function for text files with time domain data:

The command is `harm_analysis`:
```
harm_analysis --help
```
Output:
```
Usage: harm_analysis [OPTIONS] FILENAME

  Runs the harm_analysis function for a file containing time domain data

Options:
  --fs FLOAT      Sampling frequency.
  --plot          Plot the power spectrum of the data
  --sep TEXT      Separator between items.
  --sfactor TEXT  Scaling factor. The data will be multiplied by this number,
                  before the function is called. Examples: 1/8, 5, etc
  --help          Show this message and exit.
```
