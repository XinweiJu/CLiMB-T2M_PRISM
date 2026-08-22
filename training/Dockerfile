FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    numpy==2.2.6 \
    opencv-python-headless==4.11.0.86 \
    pillow==11.0.0 \
    tensorboardX==2.6.4

WORKDIR /workspace/prism_depth_climb/training
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run_three_stage.py"]
