# syntax=docker/dockerfile:1.6
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11.11
ARG HELICS_VERSION=3.6.1
ARG HELICS_GIT_COMMIT=47dd819
ARG NS3_VERSION=3.35
ARG NS3_SHA1=943a19c6b92b36d923671ae90a065f7ffb0dbabc
ARG HELICS_NS3_REF=HELICS-v3.x-waf
ARG HELICS_NS3_GIT_COMMIT=3e5879e

ENV HELICS_ROOT=/opt/helics \
    NS3_ROOT=/opt/ns-allinone-3.35/ns-3.35 \
    PATH=/opt/helics/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/ns-allinone-3.35/ns-3.35/build/lib:/opt/helics/lib:/opt/helics/lib64 \
    CMAKE_PREFIX_PATH=/opt/helics \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build pkg-config wget \
        bzip2 xz-utils libatomic1 libboost-all-dev libbz2-dev libffi-dev \
        libgdbm-dev liblzma-dev libncursesw5-dev libnss3-dev libreadline-dev \
        libsqlite3-dev libssl-dev libzmq3-dev tk-dev uuid-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
        -O /tmp/python.tgz \
    && tar -xzf /tmp/python.tgz -C /tmp \
    && cd "/tmp/Python-${PYTHON_VERSION}" \
    && ./configure --prefix=/usr/local --enable-shared --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3 \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip \
    && ldconfig \
    && rm -rf /tmp/python.tgz "/tmp/Python-${PYTHON_VERSION}"

# Compile with one job.  This is intentional for Docker Desktop/WSL2 hosts;
# unrestricted parallel C++ compilation previously exhausted the engine.
RUN git -c http.version=HTTP/1.1 clone --depth 1 --branch "v${HELICS_VERSION}" \
        https://github.com/GMLC-TDC/HELICS.git /tmp/HELICS \
    && test "$(git -C /tmp/HELICS rev-parse --short=7 HEAD)" = "${HELICS_GIT_COMMIT}" \
    && cmake -S /tmp/HELICS -B /tmp/HELICS/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${HELICS_ROOT}" \
        -DBUILD_SHARED_LIBS=ON \
        -DHELICS_BUILD_TESTS=OFF \
        -DHELICS_BUILD_EXAMPLES=OFF \
    && cmake --build /tmp/HELICS/build --parallel 1 \
    && cmake --install /tmp/HELICS/build \
    && rm -rf /tmp/HELICS \
    && ldconfig

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        "helics[cli]==${HELICS_VERSION}" \
        "numpy==2.1.3" \
        "pandas==2.2.3" \
        "pyarrow==18.1.0" \
        "scipy==1.14.1"

RUN wget -q "https://www.nsnam.org/release/ns-allinone-${NS3_VERSION}.tar.bz2" \
        -O /tmp/ns-allinone.tar.bz2 \
    && echo "${NS3_SHA1}  /tmp/ns-allinone.tar.bz2" | sha1sum -c - \
    && tar -xjf /tmp/ns-allinone.tar.bz2 -C /opt \
    && rm /tmp/ns-allinone.tar.bz2

RUN git -c http.version=HTTP/1.1 clone --depth 1 --branch "${HELICS_NS3_REF}" \
        https://github.com/GMLC-TDC/helics-ns3.git \
        "${NS3_ROOT}/contrib/helics" \
    && test "$(git -C "${NS3_ROOT}/contrib/helics" rev-parse --short=7 HEAD)" \
        = "${HELICS_NS3_GIT_COMMIT}"

COPY net_fed.cc ${NS3_ROOT}/scratch/net_fed.cc

RUN cd "${NS3_ROOT}" \
    && ./waf configure \
        --build-profile=optimized \
        --disable-python \
        --with-helics="${HELICS_ROOT}" \
        --disable-werror \
        --disable-examples \
        --disable-tests \
    && ./waf build -j1

RUN python --version \
    && helics_broker --version \
    && test -x "${NS3_ROOT}/build/scratch/net_fed" \
    && ldd "${NS3_ROOT}/build/scratch/net_fed" \
    && ! ldd "${NS3_ROOT}/build/scratch/net_fed" | grep -q "not found"

WORKDIR /workspace
CMD ["bash"]
