FROM python:3.10-slim

# Default port (can be overridden at runtime with -e NETWORK_TRANSMISSION_PORT=xxx)
ENV NETWORK_TRANSMISSION_PORT=10020
ENV HOST="0.0.0.0"

# Install system dependencies
# - iproute2: for tc (traffic control) with netem
# - opus-tools: for opusenc/opusdec
# - libopus-dev: opus library
# - libsndfile1: for soundfile
# - swig, build-essential, python3-dev: for building python-webrtc-audio-processing
# - git: for initializing submodules
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    opus-tools \
    libopus-dev \
    libsndfile1 \
    swig \
    build-essential \
    python3-dev \
    libc++-dev \
    libc++abi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/plugins/attacks/network_transmission/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build and install python-webrtc-audio-processing (clone from fork with cross-platform support)
RUN git clone https://github.com/nesovicelena/python-webrtc-audio-processing.git /tmp/python-webrtc-audio-processing && \
    cd /tmp/python-webrtc-audio-processing && \
    # Clone webrtc-audio-processing submodule (pinned to compatible commit)
    git clone https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing.git webrtc-audio-processing && \
    cd webrtc-audio-processing && git checkout ee8cfef49b8417c2f0ba65a249d2ee8c360d19ab && cd .. && \
    python setup.py build && \
    python setup.py install && \
    rm -rf /tmp/python-webrtc-audio-processing

COPY src/plugins/attacks/network_transmission .

# Expose the default port
EXPOSE 10020

# Run the FastAPI server using Uvicorn
# Note: Container must be run with --cap-add=NET_ADMIN for tc to work
CMD uvicorn app:app --host=$HOST --port=$NETWORK_TRANSMISSION_PORT
