import os
import sys
import shutil
import subprocess

def check_command(cmd):
    return shutil.which(cmd) is not None

def get_disk_space(path):
    total, used, free = shutil.disk_usage(path)
    return free // (2**30)  # GiB

def main():
    print("=== AutoML Assistant Environment Check ===\n")
    
    # 1. Check Python and Venv
    print(f"Python Version: {sys.version}")
    print(f"Virtual Env: {os.environ.get('VIRTUAL_ENV', 'None')}")
    
    # 2. Check Homebrew
    brew_path = "/opt/homebrew/bin/brew"
    if os.path.exists(brew_path):
        print(f"[OK] Homebrew found at {brew_path}")
    elif check_command("brew"):
        print("[OK] Homebrew found in PATH")
    else:
        print("[FAIL] Homebrew not found. Please install it from brew.sh")

    # 3. Check libomp
    libomp_exists = False
    possible_paths = [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            libomp_exists = True
            print(f"[OK] libomp found at {p}")
            break
    
    if not libomp_exists:
        print("[FAIL] libomp is missing. Run: brew install libomp")

    # 4. Check Disk Space
    home_free = get_disk_space(os.path.expanduser("~"))
    print(f"Free space on home drive: {home_free} GiB")
    if home_free < 5:
        print("[WARNING] Very low disk space. Please free up at least 5-10 GiB.")

    print("\nCheck complete.")

if __name__ == "__main__":
    main()
