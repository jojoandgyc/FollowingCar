ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG RKNN_WHL=.cache/rknn-toolkit2/packages/rknn_toolkit2.whl
ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_SPEC=torch==2.3.1+cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    RKNN_BACKEND=toolkit2

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN if [ -n "${TORCH_SPEC}" ]; then \
        python3 -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"; \
    fi

COPY ${RKNN_WHL} /tmp/rknn_toolkit2/
RUN python3 -m pip install --no-cache-dir /tmp/rknn_toolkit2/*.whl \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps onnx==1.16.1 \
    && rm -rf /tmp/rknn_toolkit2

WORKDIR /workspace

CMD ["python3", "-c", "import cv2, numpy; from rknn.api import RKNN; print('RKNN Toolkit2 dev image OK')"]
