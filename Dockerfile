# Use the CUDA-enabled Ubuntu base
FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

###############################################################################
# 1) Install system packages
###############################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    apt-utils \
    software-properties-common \
    ffmpeg \
    swig \
    libffi-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    gcc \
    pciutils \
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xauth \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

###############################################################################
# 2) Install Miniconda (and set up PATH)
###############################################################################
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm -f /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -afy

# So conda is found by default
ENV PATH=$CONDA_DIR/bin:$PATH

###############################################################################
# 3) Create a conda environment named "mimicplay" with Python 3.9
###############################################################################
RUN conda create --name mimicplay python=3.9 -y

# By default, Docker spawns a new shell per RUN. 
# We'll use conda's "run" so pip installs go into that env.
# This SHELL directive ensures *subsequent* RUN commands happen inside mimicplay.
SHELL ["conda", "run", "-n", "mimicplay", "/bin/bash", "-c"]

###############################################################################
# 4) Install Python packages into "mimicplay"
###############################################################################
RUN pip install --upgrade pip && \
    pip install mujoco cmake 

###############################################################################
# 5) Working directory & project installs
###############################################################################
WORKDIR /code

# Clone and install robosuite
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
    cd robosuite && \
    git checkout v1.4.1_libero && \
    pip install -r requirements.txt && \
    pip install -r requirements-extra.txt && \
    pip install -e .

# Clone and install BDDL
RUN git clone https://github.com/StanfordVL/bddl.git && \
    cd bddl && \
    pip install -e .

# Clone and install LIBERO
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && \
    cd LIBERO && \
    pip install -r requirements.txt && \
    pip install -e .

# Clone and install robomimic (mimicplay-libero branch)
RUN git clone https://github.com/ARISE-Initiative/robomimic && \
    cd robomimic && \
    git checkout mimicplay-libero && \
    pip install -e .

# Clone and install MimicPlay
RUN git clone https://github.com/j96w/MimicPlay.git && \
    cd MimicPlay && \
    pip install -e .

###############################################################################
# 6) Auto-activate mimicplay in interactive shells
###############################################################################
RUN echo "conda activate mimicplay" >> ~/.bashrc

# Start an interactive shell by default
CMD ["/bin/bash"]