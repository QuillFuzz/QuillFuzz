FROM ubuntu:24.04

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
    tmux \
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


RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ENV LLVM_SYS_140_PREFIX=/usr/lib/llvm-14
ENV PATH="/usr/lib/llvm-14/bin:${PATH}"

# Set up working directory
WORKDIR /QuillFuzz

# Create venv with uv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install build helpers and Conan (Required for tket C++ dependencies)
RUN uv pip install wheel maturin hatchling "conan>=2.0.0,<3"
RUN conan profile detect

# Explicitly add the Quantinuum Conan remote
RUN conan remote add tket https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs

# Smart Wrapper: Force Conan to build missing binaries ONLY on 'install' commands
RUN mv /opt/venv/bin/conan /opt/venv/bin/conan-real && \
    echo '#!/bin/bash\nif [[ "$1" == "install" ]]; then\n    /opt/venv/bin/conan-real "$@" --build=missing\nelse\n    /opt/venv/bin/conan-real "$@"\nfi' > /opt/venv/bin/conan && \
    chmod +x /opt/venv/bin/conan

COPY pyproject.toml .

RUN uv pip install --no-build-isolation -r pyproject.toml

WORKDIR /QuillFuzz

COPY . /QuillFuzz

# Final cleanups
RUN uv cache clean

# Set the default command
CMD ["/bin/bash"]
