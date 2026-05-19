# Use Ubuntu 24.04 (Noble) as base image to provide newer glibc (>=2.38) required by tket
FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic system dependencies including Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
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

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV LLVM_SYS_140_PREFIX=/usr/lib/llvm-14
ENV PATH="/usr/lib/llvm-14/bin:${PATH}"

# Install uv
RUN pip install uv

# Set up working directory
WORKDIR /QuillFuzz

# Copy project files
COPY pyproject.toml .

# --- Install Main Project Dependencies ---
WORKDIR /QuillFuzz

# Install build helpers
RUN uv pip install wheel maturin

# Install dependencies
# Note:
# - We use --no-build-isolation to use the system installed tools/headers
RUN uv pip install --no-build-isolation \
    pytket\
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
    selene-sim==0.2.12 \
    guppylang==0.21.13 \
    litellm \
    botocore \
    boto3 \
    coverage

# Copy the rest of the application code
COPY . /QuillFuzz

# Final cleanups
RUN uv cache clean

# Set the default command
CMD ["/bin/bash"]
