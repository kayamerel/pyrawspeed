"""Wheel smoke test: import, bundled cameras.xml, backward-compat import path.

Run against an installed pyrawspeed (this is what cibuildwheel executes in a
fresh venv with the freshly built wheel installed):

    python tests/smoke_test.py
"""

import pyrawspeed
from pyrawspeed import _pyrawspeed  # backward-compatible import path

# Bundled camera database is present in the package
assert pyrawspeed.CAMERAS_XML.is_file(), pyrawspeed.CAMERAS_XML

# Parsing the ~732 KB XML exercises the statically linked pugixml
meta = pyrawspeed.default_camera_metadata()
assert meta.has_camera("Canon", "Canon EOS 100D")
assert not meta.has_camera("Nonexistent", "Camera")

# Native types and the explicit-path constructor still work
assert isinstance(meta, _pyrawspeed.CameraMetaData)
explicit = pyrawspeed.CameraMetaData(pyrawspeed.cameras_xml_path())
assert explicit.has_camera("Canon", "Canon EOS 100D")
assert hasattr(_pyrawspeed, "decode")
assert pyrawspeed.RawImageType.UINT16 is not None
assert pyrawspeed.CFAColor.RED is not None

# Exception hierarchy sanity
assert issubclass(pyrawspeed.TiffParserError, pyrawspeed.RawParserError)
assert issubclass(pyrawspeed.RawParserError, pyrawspeed.RawspeedError)
assert issubclass(pyrawspeed.RawDecoderError, pyrawspeed.RawspeedError)

# decode() refuses a nonexistent file with a rawspeed exception, not a crash
try:
    pyrawspeed.decode("/nonexistent/file.raw")
except pyrawspeed.RawspeedError:
    pass
else:
    raise AssertionError("decode() of a missing file should raise RawspeedError")

print("pyrawspeed smoke test passed")
