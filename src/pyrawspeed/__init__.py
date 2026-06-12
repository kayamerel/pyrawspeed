"""pyrawspeed — Python bindings for the rawspeed RAW decoding library."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from . import _pyrawspeed
from ._pyrawspeed import (
    CameraMetaData,
    CFAColor,
    ColorFilterArray,
    RawImage,
    RawImageType,
    RawspeedError,
    RawDecoderError,
    FileIOError,
    CameraMetadataError,
    RawspeedIOError,
    RawParserError,
    CiffParserError,
    FiffParserError,
    TiffParserError,
)

__version__ = "0.1.1"

#: Absolute path to the cameras.xml bundled with this wheel. It is the exact
#: copy from the rawspeed commit the extension module was compiled against.
CAMERAS_XML: Path = Path(__file__).resolve().parent / "data" / "cameras.xml"


def cameras_xml_path() -> str:
    """Return the path to the bundled cameras.xml as a string."""
    if not CAMERAS_XML.is_file():
        raise FileNotFoundError(
            f"Bundled cameras.xml not found at {CAMERAS_XML}. "
            "This build of pyrawspeed was made without the bundled camera "
            "database; pass an explicit path to CameraMetaData instead."
        )
    return str(CAMERAS_XML)


@lru_cache(maxsize=1)
def default_camera_metadata() -> CameraMetaData:
    """CameraMetaData loaded from the bundled cameras.xml (cached)."""
    return CameraMetaData(cameras_xml_path())


def decode(path: str | os.PathLike[str], meta: CameraMetaData | None = None) -> RawImage:
    """Decode a RAW file.

    Uses the bundled camera database when *meta* is not given.
    """
    if meta is None:
        meta = default_camera_metadata()
    return _pyrawspeed.decode(os.fspath(path), meta)


__all__ = [
    "CAMERAS_XML",
    "CameraMetaData",
    "CFAColor",
    "ColorFilterArray",
    "RawImage",
    "RawImageType",
    "cameras_xml_path",
    "decode",
    "default_camera_metadata",
    "RawspeedError",
    "RawDecoderError",
    "FileIOError",
    "CameraMetadataError",
    "RawspeedIOError",
    "RawParserError",
    "CiffParserError",
    "FiffParserError",
    "TiffParserError",
    "_pyrawspeed",
]
