# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python bindings (nanobind) for [rawspeed](https://github.com/darktable-org/rawspeed), a fast C++ RAW image decoding library. rawspeed is vendored as a git submodule at `rawspeed/` — do not edit it. Binding code lives in `src/pyrawspeed.cpp` (a single ~155-line file); `src/pyrawspeed/__init__.py` is a thin pure-Python layer that re-exports the native symbols and adds a bundled-`cameras.xml` convenience API (`decode(path)` with optional meta, `default_camera_metadata()`, `CAMERAS_XML`).

## Build & install

```bash
# One-time setup
git submodule update --init        # fetch rawspeed
pip install scikit-build-core nanobind ninja cmake numpy

# Build + install (rebuilds the C++ extension)
pip install . --no-build-isolation
```

`--no-build-isolation` is required so pip uses the already-installed nanobind/scikit-build-core instead of fetching them in a sandbox. Any change to `src/pyrawspeed.cpp` or `CMakeLists.txt` requires re-running this install.

Build configuration is in `pyproject.toml` (`[tool.scikit-build]`: Release build, OpenMP off, stable-ABI wheels on Python ≥3.12 via `wheel.py-api = "cp312"` + `STABLE_ABI` in CMake) and the top-level `CMakeLists.txt`, which disables rawspeed's tests/benchmarks/fuzzers and builds only the library plus the `_pyrawspeed` extension module. The CMake install step also copies `rawspeed/data/cameras.xml` into the package (`pyrawspeed/data/`); `wheel.exclude` strips rawspeed/pugixml install pollution (`bin/`, `include/`, `lib/`, `share/`) out of the wheel.

## Prebuilt wheels (CI)

`.github/workflows/wheels.yml` builds wheels with cibuildwheel on every push/PR (Linux x86_64+aarch64 manylinux_2_28, macOS 14+ arm64+x86_64, Windows experimental/non-blocking) and attaches wheels + sdist to a GitHub Release on `v*` tags. Constraints baked into `[tool.cibuildwheel]`: rawspeed supports only GCC ≥12 / Clang ≥16 / Apple Clang (no MSVC — Windows uses `CC=clang`/`CXX=clang++` with the GNU driver), requires `MACOSX_DEPLOYMENT_TARGET` ≥13.5, and applies `-march=native` unless `BINARY_PACKAGE_BUILD=ON` (set for CI wheels only; local source builds keep native codegen). Test locally with `pipx run cibuildwheel --platform macos` (or `--platform linux` with Docker).

## Testing

`tests/smoke_test.py` is a self-contained smoke script (also run by cibuildwheel against every built wheel) — run it against an installed pyrawspeed:

```bash
python tests/smoke_test.py
```

It covers import, the bundled `cameras.xml`, pugixml parsing, the exception hierarchy, and the backward-compatible `from pyrawspeed import _pyrawspeed` path. There is no decode test of a real RAW file in CI (no sample file in the repo).

## Architecture

The extension module is `pyrawspeed._pyrawspeed`. `src/pyrawspeed.cpp` exposes:

- `decode(path, meta) -> RawImage` — wraps the full rawspeed pipeline in one call: `FileReader -> Buffer -> RawParser -> RawDecoder -> decodeRaw() + decodeMetaData()`.
- `RawImage` — read-only properties (`width`, `height`, `black_level`, `white_point`, `cfa`, `wb_coeffs`, `crop_offset_x/y`, ...). `.pixels` is a **zero-copy** numpy view (`uint16` or `float32`, shape `[height, width*cpp]`); a capsule holding a `RawImage` copy keeps the underlying buffer alive while the array exists. Strides are in **elements, not bytes** (rawspeed's `pitch()` is element-based; see commit ad6e89d).
- `CameraMetaData` — wrapped in `CameraMetaDataHolder` (shared_ptr) to work around an Apple Clang issue where nanobind tries to instantiate a failing copy wrapper.
- rawspeed's C++ exception hierarchy is translated to Python exceptions (`RawspeedError` base, with `RawDecoderError`, `TiffParserError`, etc. as subclasses). When exposing new rawspeed calls that can throw, register parent exceptions before children.

When adding new properties to `RawImage`, note the pattern: `RawImage` is a smart-pointer-like handle, so accessors dereference it (`(*img).field`), and optional rawspeed values (`std::optional`) are mapped to `None`.
