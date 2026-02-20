#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo ">>> 0. Securing Working Directory..."
# Find exactly where this script lives, and CD into the root of the project
# (Assuming this script is in a 'scripts' folder, this moves up one directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Working directory set to: $PROJECT_ROOT"

echo ">>> 1. Installing Miniforge (Bypasses Anaconda ToS)..."
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Downloading and installing Miniforge..."
    mkdir -p ~/miniforge3
    curl -L -o ~/miniforge3/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash ~/miniforge3/miniforge.sh -b -u -p ~/miniforge3
    rm ~/miniforge3/miniforge.sh
else
    echo "Miniforge already installed."
fi

# Initialize Conda/Mamba for the current script session
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
source "$HOME/miniforge3/etc/profile.d/mamba.sh"

echo ">>> 2. Setting up Conda Environment & Dependencies..."
if ! conda info --envs | grep -q 'quillfuzz_env'; then
    conda create -y -n quillfuzz_env python=3.11
fi

conda activate quillfuzz_env

# Install pre-compiled binaries from conda-forge
# Added pkgconfig to ensure Rust can find the C-libraries!
conda install -y \
    llvmdev=14 \
    cmake \
    graphviz \
    git \
    curl \
    compilers \
    pkgconfig \
    conan \
    gdb \
    zlib \
    libxml2 \
    ncurses \
    libffi

# Initialize Conan profile if it doesn't exist
# force=True might be needed if it exists but is wrong, but 'detect' is usually safe first run
conan profile detect --force || true

# Point to Conda's LLVM 14 installation path (Crucial for qir-runner)
export LLVM_SYS_140_PREFIX="$CONDA_PREFIX"

# Add Conda environment, Cargo, Local bin, and user local bin to PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo ">>> 3. Installing/Updating Rust..."
if ! command -v rustup &> /dev/null; then
    echo "Installing the latest Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
    echo "Updating Rust to the latest stable version..."
    source "$HOME/.cargo/env" || true
    rustup update stable
fi
source "$HOME/.cargo/env"

echo ">>> 4. Installing/Updating uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "Updating uv..."
    uv self update || true
fi

echo ">>> 5. Initializing uv Project..."
if [ ! -f "pyproject.toml" ]; then
    uv init --no-workspace
fi

# Create virtual environment EXPLICITLY using the Conda Python version
if [ ! -d ".venv" ]; then
    uv venv --python="$CONDA_PREFIX/bin/python"
fi

echo ">>> 6. Building 'qir-runner' from source..."
if [ ! -d "libs/qir-runner" ]; then
    echo "Cloning qir-runner repository..."
    rm -rf libs/qir-runner  
    mkdir -p libs
    git clone https://github.com/CQCL/qir-runner.git libs/qir-runner
else
    echo "qir-runner repository already cloned."
fi

pushd libs/qir-runner

# Keep massive temporary build files locally so we don't blow up the server's shared /tmp
export CARGO_TARGET_DIR="$PWD/target"

echo "Building qir-runner Rust components..."
cargo build --release

echo "Moving qir-runner binary to local bin..."
mkdir -p "$HOME/.local/bin"
# Verify the binary exists before moving to prevent silent failures
if [ -f "target/release/qir-runner" ]; then
    cp target/release/qir-runner "$HOME/.local/bin/"
else
    echo "ERROR: qir-runner binary was not generated!"
    exit 1
fi

echo "Installing qir-runner Python package..."
cd pip
uv pip install .

popd

echo ">>> 7. Installing Python Dependencies..."
# Ensure legacy setuptools and wheel are installed first to provide pkg_resources for build dependencies
uv pip install "setuptools<70" wheel

# Standard PyPI packages
# Using --no-build-isolation prevents the build from failing due to missing build dependencies (pkg_resources)
uv pip install --no-build-isolation \
    pytket \
    qiskit \
    pytket-qiskit \
    matplotlib \
    sympy \
    z3-solver \
    cirq \
    tket2 \
    pytket-qir \
    qnexus \
    tket \
    selene-sim \
    guppylang \
    litellm \
    coverage

uv pip install --upgrade guppylang
uv pip install git+https://github.com/CQCL/hugr-qir.git
uv pip install .

# GitHub Actions Variables Setup
if [ -n "$GITHUB_ENV" ]; then
    echo "LLVM_SYS_140_PREFIX=$LLVM_SYS_140_PREFIX" >> "$GITHUB_ENV"
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH
    echo "$HOME/.local/bin" >> $GITHUB_PATH
fi

echo ">>> 8. Cleaning up caches and build artifacts..."
conda clean -a -y
uv cache clean
rm -rf "$HOME/.cargo/registry/cache"

echo "Removing cloned repositories to save space..."
rm -rf libs/

echo ">>> Setup Complete! Double checking binaries..."
ls -la "$HOME/.local/bin/qir-runner" || echo "Warning: qir-runner not found in PATH."

echo "--------------------------------------------------------"
echo "To start working, run: conda activate quillfuzz_env"
echo "--------------------------------------------------------"