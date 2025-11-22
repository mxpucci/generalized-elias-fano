#!/usr/bin/env python3
"""
Replicates the binary loading logic used by `read_data_binary<int64_t, int64_t>`
from `src/bgef.cpp` to report the min/max values stored in a dataset.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import struct
from array import array
from math import ceil, log
from typing import Tuple


def _size_t_format() -> Tuple[str, int]:
    """Return the native struct format string and byte-width for size_t."""
    size = ctypes.sizeof(ctypes.c_size_t)
    if size == 8:
        return "@Q", size
    if size == 4:
        return "@I", size
    raise RuntimeError(f"Unsupported size_t width: {size} bytes")


def read_bgef_binary(path: str) -> Tuple[int, int, int]:
    """
    Read the dataset exactly like `read_data_binary<int64_t, int64_t>` does.

    Returns:
        A tuple of (element_count, min_value, max_value).
    """
    size_t_fmt, header_bytes = _size_t_format()
    with open(path, "rb") as handle:
        header = handle.read(header_bytes)
        if len(header) != header_bytes:
            raise ValueError("File ended before reading size_t header.")
        total_elements = struct.unpack(size_t_fmt, header)[0]

        data = array("q")
        try:
            data.fromfile(handle, total_elements)
        except EOFError as exc:
            raise ValueError("File ended before reading all int64_t elements.") from exc

    if total_elements == 0:
        raise ValueError("Dataset contains zero elements after the header.")

    return total_elements, min(data), max(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report min/max values from a .bin dataset read like src/bgef.cpp."
    )
    parser.add_argument("bin_file", help="Path to the binary dataset.")
    args = parser.parse_args()

    abs_path = os.path.abspath(args.bin_file)
    count, min_val, max_val = read_bgef_binary(abs_path)
    width = max_val - min_val + 1
    coverage = count * ceil(log(width, 2))

    print(f"File       : {abs_path}")
    print(f"Elements   : {count}")
    print(f"Min value  : {min_val}")
    print(f"Max value  : {max_val}")
    print(f"N * log2(ceil(max-min+1)): {coverage}")
    print(f"Bitpacking efficiency: {coverage / (count * 64)}")


if __name__ == "__main__":
    main()

