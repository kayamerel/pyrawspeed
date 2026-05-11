import numpy as np
from pyrawspeed import _pyrawspeed as rs

CAMERAS_XML = "/Users/kayawoo/Desktop/pyrawspeed/rawspeed/data/cameras.xml"
RAW_FILE    = "/Users/kayawoo/Desktop/epfl/cs_ba6/IVRL/Kaya_s Negatives/DSCF8338.RAF"

meta = rs.CameraMetaData(CAMERAS_XML)
img  = rs.decode(RAW_FILE, meta)

cfa_w, cfa_h = img.cfa.size
print(f"{img.make} {img.model}")
print(f"size:        {img.width} x {img.height}")
print(f"CFA:         {cfa_w}x{cfa_h} pattern")
print(f"black level: {img.black_level}")
print(f"white point: {img.white_point}")
print(f"ISO:         {img.iso_speed}")
print(f"pixels:      {img.pixels.shape}  dtype={img.pixels.dtype}")

# Normalize to [0, 1]
black = img.black_level
white = img.white_point
normalized = (img.pixels.astype(np.float32) - black) / (white - black)
normalized = np.clip(normalized, 0.0, 1.0)

print(f"\nnormalized:  min={normalized.min():.4f}  max={normalized.max():.4f}  mean={normalized.mean():.4f}")
