# Use a base image with glibc (Debian-based) to ensure compatibility with most tools
FROM python:3.11-bookworm

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libncurses5-dev \
    zlib1g-dev \
    cmake \
    graphviz \
    gdb \
    lsb-release \
    wget \
    software-properties-common \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    libclang-14-dev \
    libpolly-14-dev \
    time \
    && rm -rf /var/lib/apt/lists/*


# Set LLVM environment variables for qir-runner
ENV LLVM_SYS_140_PREFIX=/usr/lib/llvm-14
ENV PATH="/usr/lib/llvm-14/bin:$PATH"

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
RUN pip install uv

# Install Conan (C++ package manager)
RUN pip install conan && \
    conan profile detect --force

# Set up working directory
# We'll use /QuillFuzz so your prompt looks like [root@... /QuillFuzz]#
WORKDIR /QuillFuzz

# Copy project files
COPY pyproject.toml .

# --- Build qir-runner ---
# Cloning and building inside Docker to ensure consistency
WORKDIR /QuillFuzz/libs
RUN git clone https://github.com/CQCL/qir-runner.git

WORKDIR /QuillFuzz/libs/qir-runner
# Build binary
RUN cargo build --release

# Install binary to local bin (exposed in PATH)
RUN cp target/release/qir-runner /usr/local/bin/

# Install python package part of qir-runner
WORKDIR /QuillFuzz/libs/qir-runner/pip
# Using uv for fast installation
RUN uv pip install --system .

# --- Install Main Project Dependencies ---
WORKDIR /QuillFuzz

# We use --system to install into the container's global python environment, 
# which is fine for a Docker container.
# "setuptools<70" and "wheel" are installed first to fix build isolation issues
RUN uv pip install --system "setuptools<70" wheel maturin

# Install dependencies
# We use the list from setup_deps.sh. 
# Notes:
# - git+https://github.com/CQCL/hugr-qir.git is installed directly
# - We use --no-build-isolation to use the system installed tools/headers
RUN uv pip install --system --no-build-isolation \
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
    coverage \
    git+https://github.com/CQCL/hugr-qir.git

# Copy the rest of the application code
COPY . /QuillFuzz

# Final cleanups
RUN uv cache clean && \
    rm -rf /root/.cargo/registry/cache

# Set the default command
CMD ["/bin/bash"]
