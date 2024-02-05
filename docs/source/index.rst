.. Harmonic Analysis documentation master file, created by
   sphinx-quickstart on Thu Dec 28 16:46:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Harmonic Analysis's documentation!
=============================================

.. automodule:: harm_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. contents::

Introduction
------------
The harmonic analysis function uses an FFT to estimate the following parameters from a signal containing a tone:

* THD and THD+N
* Fundamental power and frequency location
* Noise power
* SNR, SINAD
* DC level
* Total integrated noise (everything except DC and the fundamental)

Installation
------------
The harm_analysis package is available via PIP install:

.. code-block:: python

   python3 -m venv pyenv
   source pyenv/bin/activate

   pip install harm_analysis

After installing the package, the harm_analysis function should be available via import:

.. code-block:: python

    from harm_analysis import harm_analysis

Documentation on how to use the function can be found here:

.. toctree::
   :maxdepth: 1

   harm_analysis

Example usage
-------------

.. plot:: tutorial/examples/run_harm_analysis.py
   :include-source:

The code above also outputs:

.. code-block::

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


The example below shows how to use the max_bw for cases where the fundamental is not
the highest spectral component in the signal. For the example, we set max_bw to
5 kHz.

.. plot:: tutorial/examples/run_shaped_noise_example.py
   :include-source:

Command line interface
----------------------

Installing the package also installs a command line interface, that allows the user to
run the function for text files with time domain data:

The command is `harm_analysis`:

.. code-block::

    harm_analysis --help

Output:

.. code-block::

    Usage: harm_analysis [OPTIONS] FILENAME

      Runs the harm_analysis function for a file containing time domain data

    Options:
      --fs FLOAT      Sampling frequency.
      --plot          Plot the power spectrum of the data
      --sep TEXT      Separator between items.
      --sfactor TEXT  Scaling factor. The data will be multiplied by this number,
                      before the function is called. Examples: 1/8, 5, etc
      --help          Show this message and exit.
