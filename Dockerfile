FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# 避免 tzdata 弹出交互界面
ENV DEBIAN_FRONTEND=noninteractive

# Set the base image to Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    clang \
    bison \
    flex \
    libreadline-dev \
    gawk \
    tcl-dev \
    libffi-dev \
    git \
    graphviz \
    xdot \
    pkg-config \
    python3 \
    libboost-system-dev \
    libboost-python-dev \
    libboost-filesystem-dev \
    cmake \
    unzip \
    wget \
    curl \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

# 克隆 Yosys 仓库
# RUN git clone https://github.com/YosysHQ/yosys.git /opt/yosys

# 编译 Yosys
# WORKDIR /opt/yosys
# RUN git submodule update --init --recursive
#
# RUN make -j$(nproc) && make install

# 设置默认命令
#CMD ["yosys", "-V"]

# 更新 pip
RUN pip3 install --upgrade pip setuptools wheel

# 先安装 PyTorch (对应 torch==2.0.1 + cu117)
RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117 \
    && pip3 install torch-geometric==2.4.0

# 安装 PyTorch Geometric 依赖
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f https://data.pyg.org/whl/torch-2.0.1+cu117.html


# 额外依赖
RUN pip3 install \
    "numpy>=1.23,<1.25" \
    scipy \
    scikit-learn \
    pandas \
    networkx \
    pyyaml \
    ogb \
    py-aiger \
    matplotlib \
    tqdm \
    ipython \
    jupyterlab \
    progress \
    einops \
    yacs \
    tensorboard
    


# 复制 requirements.txt 并安装剩余依赖
# WORKDIR /app
# COPY requirements.txt /app/
# RUN pip3 install -r requirements.txt


# Set the working directory to /app
WORKDIR /app

# 克隆 deepgate4 仓库
RUN git clone https://github.com/eiinde/DeepGate4-ICLR-25_tum.git \
    && cd DeepGate4-ICLR-25_tum \
    && git clone https://github.com/zshi0616/python-deepgate.git \
    && cd python-deepgate \
    && bash ./install.sh



# 后续操作
WORKDIR /app/DeepGate4-ICLR-25_tum 
RUN cd simulator && bash build.sh 
WORKDIR /app/DeepGate4-ICLR-25_tum 
RUN mkdir -p tmp

# 执行模拟shell脚本
# RUN bash 

# 复制脚本到工作目录
# WORKDIR /app
# COPY cada1019_alpha.sh /app/cada1019_alpha
# COPY cada1019_alpha /app/



# # Make the script executable
# RUN chmod +x /app/cada1019_alpha
