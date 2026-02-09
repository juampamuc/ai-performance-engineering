FROM nvcr.io/nvidia/pytorch:26.01-py3@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    libboost-program-options-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip "setuptools<81" wheel

# Keep utility deps aligned with the open decoupled image.
RUN python -m pip install numpy matplotlib

# Pin DeepGEMM to the legacy reference commit used by old_container.
RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM \
    && cd /opt/DeepGEMM \
    && git checkout 0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51 \
    && git submodule update --init --recursive \
    && python -m pip uninstall -y deep_gemm || true \
    && python -m pip install --no-build-isolation /opt/DeepGEMM

# nvbandwidth (open-source) for dedicated bandwidth bundle.
RUN git clone --depth 1 https://github.com/NVIDIA/nvbandwidth.git /opt/nvbandwidth \
    && cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/nvbandwidth/build -j"$(nproc)" \
    && cp /opt/nvbandwidth/build/nvbandwidth /usr/local/bin/nvbandwidth

WORKDIR /workspace

CMD ["bash"]
