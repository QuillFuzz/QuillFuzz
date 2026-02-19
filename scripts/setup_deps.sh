#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo ">>> 1a. Installing/Activating Homebrew locally..."

# Check if the local brew executable exists, rather than relying on the global PATH
if [ ! -f "$HOME/.linuxbrew/bin/brew" ]; then
    echo "Homebrew not found. Installing to ~/.linuxbrew without sudo..."
    
    # Ensure the directory is completely clear before cloning
    rm -rf ~/.linuxbrew/Homebrew
    
    git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew
    mkdir -p ~/.linuxbrew/bin
    ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin
else
    echo "Homebrew installation found at ~/.linuxbrew."
fi

# Always evaluate the shellenv so the 'brew' command works for the rest of this script
eval "$($HOME/.linuxbrew/bin/brew shellenv)"

echo ">>> 1b. Installing System Dependencies via Homebrew..."
# Prevent brew from auto-updating everything every time you run the script
export HOMEBREW_NO_AUTO_UPDATE=1

brew install \
    llvm@14 \
    cmake \
    graphviz \
    git \
    curl \
    gcc \
    python@3.11 \
    gdb \
    zlib \
    libxml2 \
    ncurses \
    libffi

echo ">>> 2. Setting up Environment Variables..."

# Point to Homebrew's LLVM 14 installation path (Crucial for qir-runner)
export LLVM_SYS_140_PREFIX="$(brew --prefix llvm@14)"

# IMPORTANT: Homebrew installs tools like zlib, libxml2, and llvm as "keg-only" 
# (meaning they aren't globally symlinked to prevent conflicts). 
# We must explicitly tell the C/Rust compilers where to find these headers and libraries.
export LDFLAGS="-L$(brew --prefix zlib)/lib -L$(brew --prefix libxml2)/lib -L$(brew --prefix libffi)/lib -L$(brew --prefix llvm@14)/lib"
export CPPFLAGS="-I$(brew --prefix zlib)/include -I$(brew --prefix libxml2)/include -I$(brew --prefix libffi)/include -I$(brew --prefix llvm@14)/include"
export PKG_CONFIG_PATH="$(brew --prefix zlib)/lib/pkgconfig:$(brew --prefix libxml2)/lib/pkgconfig:$(brew --prefix libffi)/lib/pkgconfig"

# Add LLVM, Cargo, and Local bin to PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$(brew --prefix llvm@14)/bin:$PATH"


echo ">>> 3. Installing Rust (if missing)..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed."
fi

echo ">>> 4. Installing uv (if missing)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

echo ">>> 5. Initializing uv Project..."
# Initialize only if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    uv init --no-workspace
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

echo ">>> 6. Building 'qir-runner' from source..."
# qir-runner is special; it requires a manual build step before python installation
# Check for Cargo.toml to ensure repo is actually cloned
if [ ! -d "libs/qir-runner" ]; then
    echo "Cloning qir-runner repository..."
    rm -rf libs/qir-runner  # Remove any empty/incomplete directory
    mkdir -p libs
    git clone https://github.com/CQCL/qir-runner.git libs/qir-runner
else
    echo "qir-runner repository already cloned."
fi

pushd libs/qir-runner

# Build the Rust binary (warning about lifetime is harmless)
echo "Building qir-runner Rust components..."
cargo build --release

echo "Moving qir-runner binary to local bin..."
mkdir -p "$HOME/.local/bin"
cp target/release/qir-runner "$HOME/.local/bin/"

# Install the Python package directly into the uv environment
echo "Installing qir-runner Python package..."
cd pip
uv pip install .

popd

echo ">>> 7. Installing Python Dependencies..."
# Use uv pip install to avoid workspace conflicts
# This installs directly into the virtual environment

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

# C. Install local packages
uv pip install .

# If running in GitHub Actions, save these variables for future steps
if [ -n "$GITHUB_ENV" ]; then
    echo "LLVM_SYS_140_PREFIX=$LLVM_SYS_140_PREFIX" >> "$GITHUB_ENV"
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH
    echo "$HOME/.local/bin" >> $GITHUB_PATH
fi

echo ">>> 8. Cleaning up caches and build artifacts..."

# Clean Homebrew cache
export HOMEBREW_NO_AUTO_UPDATE=1
brew cleanup

# Clean uv cache
uv cache clean

# Completely remove the cloned source code and all its build artifacts
echo "Removing cloned repositories..."
rm -rf libs/

echo ">>> Setup Complete!"