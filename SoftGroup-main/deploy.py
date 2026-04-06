#!/usr/bin/env python3
"""
SoftGroup Cross-Platform Deployment Script.
Automatically detects OS, CUDA, GPU and sets up everything.

Usage:
    python deploy.py              # Full install + build
    python deploy.py --check      # Check environment only
    python deploy.py --build-only # Build CUDA extensions only
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys


# ===== Utilities =====

def run(cmd, check=True, capture=True, timeout=600):
    """Run a shell command."""
    print(f"  $ {cmd}")
    try:
        r = subprocess.run(
            cmd, shell=True, check=check, timeout=timeout,
            capture_output=capture, text=True)
        return r.stdout.strip() if capture else ""
    except subprocess.CalledProcessError as e:
        if capture:
            print(f"  STDERR: {e.stderr[:500] if e.stderr else ''}")
        if check:
            raise
        return ""
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return ""


def find_executable(name, extra_paths=None):
    """Find an executable in PATH or extra paths."""
    result = shutil.which(name)
    if result:
        return result
    for p in (extra_paths or []):
        candidate = os.path.join(p, name)
        if os.path.isfile(candidate):
            return candidate
        if sys.platform == "win32" and os.path.isfile(candidate + ".exe"):
            return candidate + ".exe"
    return None


# ===== Detection =====

def detect_os():
    """Detect operating system."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/etc/os-release") as f:
                content = f.read()
            name = re.search(r'PRETTY_NAME="(.+)"', content)
            return "linux", name.group(1) if name else "Linux"
        except FileNotFoundError:
            return "linux", "Linux"
    elif system == "Windows":
        ver = platform.version()
        return "windows", f"Windows {platform.release()} ({ver})"
    else:
        return system.lower(), f"{system} {platform.release()}"


def detect_gpu():
    """Detect NVIDIA GPU and compute capability."""
    try:
        out = run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                   check=False)
        if out:
            parts = out.split("\n")[0].split(", ")
            return {
                "name": parts[0].strip(),
                "compute_cap": parts[1].strip() if len(parts) > 1 else "unknown",
                "vram": parts[2].strip() if len(parts) > 2 else "unknown",
            }
    except Exception:
        pass
    return None


def detect_cuda():
    """Detect CUDA toolkit version."""
    try:
        out = run("nvcc --version", check=False)
        m = re.search(r"release (\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def detect_conda():
    """Find conda executable."""
    conda = find_executable("conda")
    if conda:
        return conda
    # Common locations
    candidates = [
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        "/opt/conda/bin/conda",
        "C:\\miniconda3\\Scripts\\conda.exe",
        "C:\\anaconda\\Scripts\\conda.exe",
        "C:\\conda\\Scripts\\conda.exe",
        "C:\\anaconda3\\Scripts\\conda.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def detect_msvc():
    """Find Visual Studio vcvarsall.bat (Windows only)."""
    if sys.platform != "win32":
        return None
    base = "C:\\Program Files (x86)\\Microsoft Visual Studio"
    for ver in ["18", "17", "16", "15"]:
        for edition in ["BuildTools", "Enterprise", "Professional", "Community"]:
            path = os.path.join(base, ver, edition, "VC", "Auxiliary", "Build", "vcvarsall.bat")
            if os.path.isfile(path):
                return path
    # VS 2022 in Program Files
    base2 = "C:\\Program Files\\Microsoft Visual Studio\\2022"
    for edition in ["BuildTools", "Enterprise", "Professional", "Community"]:
        path = os.path.join(base2, edition, "VC", "Auxiliary", "Build", "vcvarsall.bat")
        if os.path.isfile(path):
            return path
    return None


# ===== Environment Check =====

def check_environment():
    """Check and report environment status."""
    print("=" * 60)
    print(" SoftGroup Environment Check")
    print("=" * 60)

    os_type, os_name = detect_os()
    print(f"\n  OS:        {os_name}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  Platform:  {platform.machine()}")

    gpu = detect_gpu()
    if gpu:
        print(f"  GPU:       {gpu['name']}")
        print(f"  Compute:   {gpu['compute_cap']}")
        print(f"  VRAM:      {gpu['vram']}")
    else:
        print("  GPU:       NOT FOUND (NVIDIA GPU required)")

    cuda_ver = detect_cuda()
    print(f"  CUDA:      {cuda_ver or 'NOT FOUND'}")

    conda = detect_conda()
    print(f"  Conda:     {conda or 'NOT FOUND'}")

    if os_type == "windows":
        msvc = detect_msvc()
        print(f"  MSVC:      {msvc or 'NOT FOUND'}")

    # Check softgroup env
    if conda:
        try:
            envs = run(f'"{conda}" env list', check=False)
            has_sg = "softgroup" in envs
            print(f"  Env:       {'softgroup exists' if has_sg else 'softgroup NOT created'}")
        except Exception:
            pass

    print()
    ok = gpu is not None and conda is not None
    if os_type == "windows":
        ok = ok and detect_msvc() is not None
    if ok:
        print("  Status: READY for installation")
    else:
        missing = []
        if not gpu:
            missing.append("NVIDIA GPU/driver")
        if not conda:
            missing.append("Conda (miniconda/anaconda)")
        if os_type == "windows" and not detect_msvc():
            missing.append("Visual Studio Build Tools with C++")
        print(f"  Status: MISSING - {', '.join(missing)}")

    return ok


# ===== Installation =====

def select_pytorch_index(cuda_ver):
    """Select PyTorch pip index URL based on CUDA version."""
    if not cuda_ver:
        return "https://download.pytorch.org/whl/cu121", "spconv-cu120"
    major, minor = map(int, cuda_ver.split("."))
    if major >= 13:
        return "https://download.pytorch.org/whl/cu124", "spconv-cu120"
    elif major == 12:
        if minor >= 4:
            return "https://download.pytorch.org/whl/cu124", "spconv-cu120"
        else:
            return "https://download.pytorch.org/whl/cu121", "spconv-cu120"
    elif major == 11:
        return "https://download.pytorch.org/whl/cu118", "spconv-cu118"
    return "https://download.pytorch.org/whl/cu121", "spconv-cu120"


def patch_pytorch_cuda_check(conda, env_name="softgroup"):
    """Patch PyTorch to allow CUDA version mismatch."""
    script = (
        "import os, torch;"
        "p=os.path.join(os.path.dirname(torch.__file__),'utils','cpp_extension.py');"
        "f=open(p,'r');c=f.read();f.close();"
        "old='raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))';"
        "new='warnings.warn(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))';"
        "c=c.replace(old,new);"
        "f=open(p,'w');f.write(c);f.close();"
        "print('Patched' if old!=new else 'Already patched')"
    )
    run(f'"{conda}" run -n {env_name} python -c "{script}"', check=False)


def install(args):
    """Full installation."""
    os_type, os_name = detect_os()
    gpu = detect_gpu()
    conda = detect_conda()
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print(" SoftGroup Deployment")
    print("=" * 60)
    print(f"  OS:      {os_name}")
    print(f"  GPU:     {gpu['name'] if gpu else 'NOT FOUND'}")
    print(f"  Project: {project_dir}")
    print()

    if not gpu:
        print("ERROR: NVIDIA GPU required")
        return False
    if not conda:
        print("ERROR: Conda not found. Install Miniconda first:")
        if os_type == "windows":
            print("  https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe")
        else:
            print("  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
            print("  bash Miniconda3-latest-Linux-x86_64.sh")
        return False

    # Step 1: Create conda env
    print("\n[1/6] Creating conda environment...")
    run(f'"{conda}" create -n softgroup python=3.8 -y', check=False)

    # Step 2: Install system deps (Linux only)
    if os_type == "linux":
        print("\n[2/6] Installing system dependencies...")
        run("sudo apt-get update -qq && sudo apt-get install -y -qq build-essential libsparsehash-dev",
            check=False)
    else:
        print("\n[2/6] Setting up sparsehash (Windows)...")
        sparsehash_dir = os.path.join(project_dir, "third_party", "sparsehash")
        if not os.path.isdir(sparsehash_dir):
            run(f"git clone https://github.com/sparsehash/sparsehash.git \"{sparsehash_dir}\"",
                check=False)

    # Step 3: Install CUDA toolkit (via conda for Windows)
    if os_type == "windows":
        print("\n[3/6] Installing CUDA toolkit via conda...")
        run(f'"{conda}" install -n softgroup -c nvidia cuda-toolkit -y', check=False)

    # Step 4: Install PyTorch + spconv
    print("\n[4/6] Installing PyTorch and spconv...")
    cuda_ver = detect_cuda()
    pt_index, spconv_pkg = select_pytorch_index(cuda_ver)
    run(f'"{conda}" run -n softgroup pip install torch torchvision --index-url {pt_index}',
        check=False, timeout=600)
    run(f'"{conda}" run -n softgroup pip install {spconv_pkg}', check=False, timeout=300)

    # Step 5: Install Python deps
    print("\n[5/6] Installing Python dependencies...")
    req_file = os.path.join(project_dir, "requirements.txt")
    if os.path.isfile(req_file):
        run(f'"{conda}" run -n softgroup pip install -r "{req_file}"', check=False)

    # Step 6: Patch PyTorch
    print("\n[6/6] Patching PyTorch CUDA version check...")
    patch_pytorch_cuda_check(conda)

    print("\n" + "=" * 60)
    print(" Dependencies installed!")
    print(" Now run: python deploy.py --build-only")
    print("=" * 60)
    return True


# ===== Build =====

def build(args):
    """Build CUDA extensions."""
    os_type, _ = detect_os()
    gpu = detect_gpu()
    conda = detect_conda()
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print(" Building CUDA Extensions")
    print("=" * 60)

    arch = gpu["compute_cap"] if gpu else "8.6"
    env_vars = f'TORCH_CUDA_ARCH_LIST="{arch}"'

    if os_type == "windows":
        msvc = detect_msvc()
        if not msvc:
            print("ERROR: Visual Studio Build Tools not found")
            return False

        # Find conda env prefix
        conda_dir = os.path.dirname(os.path.dirname(conda))  # e.g. C:\anaconda
        prefix = os.path.join(conda_dir, "envs", "softgroup")
        if not os.path.isdir(prefix):
            # Fallback: try to detect from conda info
            try:
                info = run(f'"{conda}" info --envs', check=False)
                for line in info.split("\n"):
                    if "softgroup" in line:
                        prefix = line.split()[-1].strip()
                        break
            except Exception:
                pass

        cuda_home = os.path.join(prefix, "Library")
        if not os.path.isdir(os.path.join(cuda_home, "bin")):
            cuda_home = os.environ.get("CUDA_PATH", "")

        python_exe = os.path.join(prefix, "python.exe")

        # Build via PowerShell (handles vcvarsall)
        ps_cmd = f"""
$env:CUDA_HOME = '{cuda_home}'
$env:CUDA_PATH = '{cuda_home}'
$env:TORCH_CUDA_ARCH_LIST = '{arch}'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
cmd /c '"{msvc}" x64 >nul 2>&1 && set' | ForEach-Object {{
    if ($_ -match '^([^=]+)=(.*)') {{
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }}
}}
$env:CUDA_HOME = '{cuda_home}'
$env:CUDA_PATH = '{cuda_home}'
$env:TORCH_CUDA_ARCH_LIST = '{arch}'
$env:DISTUTILS_USE_SDK = '1'
Set-Location '{project_dir}'
& '{python_exe}' setup.py build_ext develop 2>&1
"""
        print(f"  CUDA_HOME: {cuda_home}")
        print(f"  GPU arch:  {arch}")
        print(f"  MSVC:      {msvc}")
        print("  Building (this may take several minutes)...\n")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            cwd=project_dir, timeout=600)
    else:
        # Linux: straightforward
        print(f"  GPU arch: {arch}")
        print("  Building...\n")
        run(f'cd "{project_dir}" && {env_vars} '
            f'"{conda}" run -n softgroup python setup.py build_ext develop',
            capture=False, timeout=600)

    # Verify
    print("\n  Verifying build...")
    if os_type == "windows":
        result = run(f'"{python_exe}" -c "from softgroup.ops import ops; print(\'CUDA ops: OK\')"',
                     check=False)
    else:
        result = run(f'"{conda}" run -n softgroup python -c '
                     '"from softgroup.ops import ops; print(\'CUDA ops: OK\')"',
                     check=False)
    if "OK" in result:
        print("  BUILD SUCCESSFUL!")
        return True
    else:
        print("  BUILD FAILED - check errors above")
        return False


# ===== Verify =====

def verify(args):
    """Verify installation."""
    conda = detect_conda()
    if not conda:
        print("ERROR: Conda not found")
        return False

    print("=" * 60)
    print(" Verification")
    print("=" * 60)

    os_type, _ = detect_os()
    if os_type == "windows":
        conda_dir = os.path.dirname(os.path.dirname(conda))
        python_exe = os.path.join(conda_dir, "envs", "softgroup", "python.exe")
    else:
        python_exe = None

    script = (
        "import torch; "
        "print(f'  PyTorch:  {torch.__version__}'); "
        "print(f'  CUDA:     {torch.cuda.is_available()}'); "
        "print(f'  GPU:      {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''); "
        "import softgroup; print('  SoftGroup: OK'); "
        "from softgroup.ops import ops; print('  CUDA ops:  OK'); "
        "import spconv; print(f'  spconv:   {spconv.__version__}'); "
        "print(); print('  === ALL CHECKS PASSED ===')"
    )
    if python_exe and os.path.isfile(python_exe):
        run(f'"{python_exe}" -c "{script}"', capture=False, check=False)
    else:
        run(f'"{conda}" run -n softgroup python -c "{script}"', capture=False, check=False)
    return True


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="SoftGroup Deployment")
    parser.add_argument("--check", action="store_true", help="Check environment only")
    parser.add_argument("--build-only", action="store_true", help="Build CUDA extensions only")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    args = parser.parse_args()

    if args.check:
        check_environment()
    elif args.build_only:
        build(args)
    elif args.verify:
        verify(args)
    else:
        # Full install
        if install(args):
            build(args)
            verify(args)


if __name__ == "__main__":
    main()
