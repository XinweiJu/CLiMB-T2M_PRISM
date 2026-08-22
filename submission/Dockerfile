ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/tmp/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir opencv-python-headless==4.11.0.86

WORKDIR /opt/track2map-climb
COPY vendor/co-tracker /opt/models/co-tracker
COPY vendor/PRISM-CLiMB /opt/models/PRISM-CLiMB
COPY climb_io.py monocular_vo.py track2map_frontend.py trajectory_smoothing.py run.py /opt/track2map-climb/

ENTRYPOINT ["python", "/opt/track2map-climb/run.py"]
CMD ["--profile", "v3", "--input", "/input", "--output", "/output", "--backend", "track2map", "--num-runs", "5"]
