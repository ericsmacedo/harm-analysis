
Introduction
------------
The harmonic analysis function uses an FFT to estimate the following parameters from a signal containing a tone:

* THD and THD+N
* Fundamental power and frequency location
* Noise power
* SNR, SINAD
* DC level
* Total integrated noise (everything except DC and the fundamental)

The full documentation is hosted on ReadTheDocs:`Harmonic Analysis <https://harm-analysis.readthedocs.io/en/latest/index.html>`_.

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

Documentation on how to use the function can be found `here <https://harm-analysis.readthedocs.io/en/latest/harm_analysis.html>`_.

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
