# pyrawspeed

Python bindings for [rawspeed](https://github.com/darktable-org/rawspeed), a fast RAW image decoding library.

### Setup guide

### Python

Python 3.10 or newer is required. I recommend to use a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

### Python build dependencies

```
pip install scikit-build-core nanobind ninja cmake numpy
```

## Installation

After installing the requirements above:

```
pip install . --no-build-isolation
```

> **Note:** `--no-build-isolation` is required so that pip uses the nanobind
> and scikit-build-core already installed in your environment rather than
> trying to fetch them in an isolated build sandbox.