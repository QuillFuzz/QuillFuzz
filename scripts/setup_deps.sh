#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

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

# Initialize Conda for the current script session
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
source "$HOME/miniforge3/etc/profile.d/mamba.sh" # Miniforge includes mamba, a faster resolver

echo ">>> 2. Setting up Conda Environment & Dependencies..."
# Create an environment called 'quillfuzz_env' if it doesn't exist
if ! conda info --envs | grep -q 'quillfuzz_env'; then
    conda create -y -n quillfuzz_env python=3.11
fi

conda activate quillfuzz_env

# Install pre-compiled binaries from conda-forge
conda install -y \
    llvmdev=14 \
    cmake \
    graphviz \
    git \
    curl \
    compilers \
    gdb \
    zlib \
    libxml2 \
    ncurses \
    libffi

# Point to Conda's LLVM 14 installation path (Crucial for qir-runner)
export LLVM_SYS_140_PREFIX="$CONDA_PREFIX"

# Add Conda environment, Cargo, and Local bin to PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo ">>> 4. Installing Rust (if missing)..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed."
fi

echo ">>> 5. Installing uv (if missing)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

echo ">>> 6. Initializing uv Project..."
# Initialize only if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    uv init --no-workspace
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

echo ">>> 7. Building 'qir-runner' from source..."
if [ ! -d "libs/qir-runner" ]; then
    echo "Cloning qir-runner repository..."
    rm -rf libs/qir-runner  
    mkdir -p libs
    git clone https://github.com/CQCL/qir-runner.git libs/qir-runner
else
    echo "qir-runner repository already cloned."
fi

pushd libs/qir-runner

# Build the Rust binary
echo "Building qir-runner Rust components..."
cargo build --release

# RESCUE THE BINARY: Copy it to ~/.local/bin before we delete the libs folder later
echo "Moving qir-runner binary to local bin..."
mkdir -p "$HOME/.local/bin"
cp target/release/qir-runner "$HOME/.local/bin/"

# Install the Python package directly into the uv environment
echo "Installing qir-runner Python package..."
cd pip
uv pip install .

popd

echo ">>> 8. Installing Python Dependencies..."
# A. Standard PyPI packages
uv pip install \
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

# Ensure guppylang is the latest version
uv pip install --upgrade guppylang

# B. Git dependencies
uv pip install git+https://github.com/CQCL/hugr-qir.git

# C. Install local packages (NO -e flag, so we can safely clean up the folder later)
uv pip install .

# If running in GitHub Actions, save these variables for future steps
if [ -n "$GITHUB_ENV" ]; then
    echo "LLVM_SYS_140_PREFIX=$LLVM_SYS_140_PREFIX" >> "$GITHUB_ENV"
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH
    echo "$HOME/.local/bin" >> $GITHUB_PATH
fi

echo ">>> 9. Cleaning up caches and build artifacts..."

# Clean Conda cache (removes downloaded tarballs)
conda clean -a -y

# Clean uv cache
uv cache clean

# Clean global Cargo registry cache
rm -rf "$HOME/.cargo/registry/cache"

# Completely remove the cloned source code and all its massive build artifacts
echo "Removing cloned repositories..."
rm -rf libs/

echo ">>> Setup Complete! Run 'conda activate qir_env' to start working."