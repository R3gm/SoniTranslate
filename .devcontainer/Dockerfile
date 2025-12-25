FROM mcr.microsoft.com/devcontainers/python:3.10-bullseye

ARG DEBIAN_FRONTEND=noninteractive

# System deps for audio/video and common ML libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
        libportaudio2 \
        libasound2 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
