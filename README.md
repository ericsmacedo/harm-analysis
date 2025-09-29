
[![PyPI Version](https://badge.fury.io/py/harm-analysis.svg)](https://badge.fury.io/py/harm-analysis)
[![Python Build](https://github.com/ericsmacedo/harm-analysis/actions/workflows/main.yml/badge.svg)](https://github.com/ericsmacedo/harm-analysis/actions/workflows/main.yml)
[![Documentation](https://readthedocs.org/projects/harm-analysis/badge/?version=stable)](https://harm-analysis.readthedocs.io/en/stable/)
[![Coverage Status](https://coveralls.io/repos/github/ericsmacedo/harm-analysis/badge.svg?branch=main)](https://coveralls.io/github/ericsmacedo/harm-analysis?branch=main)
[![python-versions](https://img.shields.io/pypi/pyversions/harm-analysis.svg)](https://pypi.python.org/pypi/harm-analysis)
[![semantic-versioning](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/)

[![Downloads](https://img.shields.io/pypi/dm/harm-analysis.svg?label=pypi%20downloads)](https://pypi.python.org/pypi/harm-analysis)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
[![Issues](https://img.shields.io/github/issues/ericsmacedo/harm-analysis)](https://github.com/ericsmacedo/harm-analysis/issues)

# Harmonic Analysis package

* [Documentation](https://harm-analysis.readthedocs.io/en/stable/)
* [PyPI](https://pypi.org/project/harm-analysis/)
* [Sources](https://github.com/ericsmacedo/harm-analysis)
* [Issues](https://github.com/ericsmacedo/harm-analysis/issues)

## Introduction

The harmonic analysis package uses FFT to estimate parameters of an input signal.
The package provides two main functions:

- `harm_analysis`: for simulations with an injected tone, returning SNR, THDN, Noise, etc.
- `spec_analysis`: for cases without an injected tone, that auto-detects DC, tones, and noise from the spectrum.

See [usage](./docs/usage.md) for examples.


## Installation

Installing it is pretty easy:

```bash
pip install harm-analysis
```
