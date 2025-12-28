FROM mcr.microsoft.com/devcontainers/python:3.10-bullseye

ARG DEBIAN_FRONTEND=noninteractive

# System deps for audio/video and common ML libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        gnupg \
        build-essential \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev \
        libsndfile1 \
        libportaudio2 \
        libasound2 \
        libgl1 \
        libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @openai/codex@alpha \
    && rm -rf /var/lib/apt/lists/*
