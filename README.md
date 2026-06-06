# pyrawspeed

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with nanobind](https://img.shields.io/badge/built%20with-nanobind-orange)](https://github.com/wjakob/nanobind)

Python bindings for [rawspeed](https://github.com/darktable-org/rawspeed), the C++ RAW image decoding library that powers [darktable](https://www.darktable.org/).

`rawspeed` is **fast**. That speed is exactly why we built this wrapper: existing Python options were too slow for our processing pipelines, while rawspeed decodes RAW files in a fraction of the time. `pyrawspeed` exposes that performance directly to Python, returning the decoded sensor data as a **zero-copy** numpy array along with the metadata (black/white levels, CFA pattern, white balance coefficients, ...).

This wrapper was developed as part of [**ColorHead**](https://colorhead.pages.dev/), a tool for color negative film inversion, a project at the [IVRL lab](https://www.epfl.ch/labs/ivrl/) ([GitHub](https://github.com/IVRL)) at EPFL.

## Setup guide

### Requirements

- Python 3.10 or newer
- A C++ compiler toolchain (Xcode Command Line Tools on macOS, `build-essential` on Debian/Ubuntu) — the rawspeed library is compiled from source during installatio

### Install via pip from GitHub

```
pip install "git+https://github.com/kayamerel/pyrawspeed.git"
```

## Usage

```python
import numpy as np
from pyrawspeed import _pyrawspeed as rs

meta = rs.CameraMetaData("path/to/rawspeed/data/cameras.xml")
img  = rs.decode("path/to/photo.RAF", meta)

print(f"{img.make} {img.model}")
print(f"size:        {img.width} x {img.height}")
print(f"CFA:         {img.cfa.size[0]}x{img.cfa.size[1]} pattern")
print(f"black level: {img.black_level}")
print(f"white point: {img.white_point}")
print(f"ISO:         {img.iso_speed}")
print(f"pixels:      {img.pixels.shape}  dtype={img.pixels.dtype}")

# Normalize the raw data to [0, 1]
normalized = (img.pixels.astype(np.float32) - img.black_level) \
             / (img.white_point - img.black_level)
normalized = np.clip(normalized, 0.0, 1.0)
```

### API

- `CameraMetaData(cameras_xml)` — loads the camera database. Also exposes
  `has_camera(make, model, mode="")`.
- `decode(path, meta) -> RawImage` — runs the full rawspeed pipeline
  (file read → parse → decode raw data + metadata) and returns a `RawImage`.

`RawImage` properties (all read-only):

| Property | Description |
| --- | --- |
| `pixels` | **Zero-copy** numpy view of the decoded data, shape `[height, width*cpp]`, dtype `uint16` or `float32`. The view keeps the underlying buffer alive. |
| `width`, `height` | Image dimensions in pixels |
| `cpp` | Components per pixel |
| `data_type` | `RawImageType.UINT16` or `RawImageType.F32` |
| `is_cfa`, `cfa` | Whether the sensor has a color filter array, and the CFA itself (`size`, `get_color_at(x, y)`, `as_string()`) |
| `black_level`, `white_point` | Sensor black/white levels (`white_point` may be `None`) |
| `crop_offset_x`, `crop_offset_y` | Offset of the active (cropped) area within the full sensor frame |
| `make`, `model`, `iso_speed` | Camera metadata |
| `wb_coeffs` | As-shot white balance coefficients as a 4-tuple, or `None` |

Errors from rawspeed are raised as Python exceptions: `RawspeedError` is the
base class, with subclasses such as `RawDecoderError`, `TiffParserError`,
`FileIOError`, and `CameraMetadataError`.