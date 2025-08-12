#!/usr/bin/env python3
import subprocess
import os

BINARY_NAME = "my_binary"  # Change this to your executable name

print("🚀 Starting CentOS 7 build in Docker (vault repo fix)...")

# Absolute path to your project directory
project_dir = os.path.abspath(os.path.dirname(__file__))

# The shell commands to run inside Docker
build_script = f"""
set -e
# Fix CentOS 7 repo to use vault
sed -i 's|mirrorlist=|#mirrorlist=|g' /etc/yum.repos.d/CentOS-*.repo
sed -i 's|#baseurl=http://mirror.centos.org/centos/$releasever|baseurl=http://vault.centos.org/7.9.2009|g' /etc/yum.repos.d/CentOS-*.repo

yum -y install epel-release
yum -y groupinstall "Development Tools"
yum -y install cmake3
ln -s /usr/bin/cmake3 /usr/bin/cmake || true

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=skylake-avx512 -mtune=skylake-avx512 -static-libstdc++ -static-libgcc -flto" ..
cmake --build . -- -j$(nproc)
strip {BINARY_NAME}
chown {os.getuid()}:{os.getgid()} {BINARY_NAME}
"""

docker_command = [
    "docker", "run", "--rm", "-i",
    "--platform", "linux/amd64",
    "-v", f"{project_dir}:/src",
    "-w", "/src",
    "centos:7",
    "/bin/bash", "-c", build_script
]

try:
    subprocess.run(docker_command, check=True)
    print(f"✅ Build complete! Binary is in {project_dir}/build/{BINARY_NAME}")
except subprocess.CalledProcessError as e:
    print("❌ Build failed.")
    raise SystemExit(e.returncode)
