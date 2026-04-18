import os
import matplotlib
from matplotlib import style
from matplotlib.style.core import BASE_LIBRARY_PATH, read_style_directory
from matplotlib import _rc_params_in_file

print(f"Base library path: {BASE_LIBRARY_PATH}")

def debug_read_style_directory(style_dir):
    styles = {}
    for path in style_dir.glob("*.mplstyle"):
        print(f"Reading: {path}")
        try:
            styles[path.stem] = _rc_params_in_file(path)
        except UnicodeDecodeError as e:
            print(f"!!! Error reading {path}: {e}")
            # Let's see the content of the file around the error
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                    print(f"Byte at position 37: {hex(content[37]) if len(content) > 37 else 'N/A'}")
                    print(f"Context: {content[max(0, 37-20):min(len(content), 37+20)]}")
            except Exception as e2:
                print(f"Could not read as binary: {e2}")

if __name__ == "__main__":
    from pathlib import Path
    debug_read_style_directory(Path(BASE_LIBRARY_PATH))
