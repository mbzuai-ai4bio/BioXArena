# 1. Base image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# 2. Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# # 3. Replace with Alibaba Cloud mirror (optional, for faster downloads in China; remove if not needed)
# RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
#     sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 4. Install tools required for adding PPAs and common utilities
RUN apt-get update && apt-get install -y \
    software-properties-common \
    vim \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 5. Add deadsnakes PPA (contains various Python versions)
RUN add-apt-repository ppa:deadsnakes/ppa

# 6. Install Python 3.12 and its development libraries
# Note: python3.12-dev and python3.12-venv are critical for subsequent pip package installations
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# 7. Install pip (specifically for Python 3.12)
# Ubuntu's python3-pip package usually corresponds to the system default 3.10, so we use a script to install the latest pip for 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 8. Set the default Python version (modify symlinks)
# This way, typing python or python3 will automatically point to python3.12
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# # 9. Configure pip to use Tsinghua mirror by default (optional, highly recommended for China)
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 10. Set working directory
WORKDIR /workspace
