FROM python:3.7

## Install TVM and related dependencies

# TVM Deps on apt
RUN apt update && apt install -y --no-install-recommends git libgtest-dev cmake wget unzip libtinfo-dev libz-dev \
     libcurl4-openssl-dev libopenblas-dev g++ sudo python3-dev

# Manually add LLVM
RUN echo deb http://apt.llvm.org/buster/ llvm-toolchain-buster-8 main \
     >> /etc/apt/sources.list.d/llvm.list && \
     wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
     apt-get update && apt-get install -y llvm-8

# Build Gus's version of TVM
RUN cd /root && git clone --recursive https://github.com/apache/incubator-tvm tvm
WORKDIR /root/tvm
RUN git fetch
RUN git checkout 84d1eec39a10c559cb659d3f411cacce08cfdaff
RUN git submodule sync && git submodule update
RUN echo 'set(USE_LLVM llvm-config-8)' >> config.cmake
RUN echo 'set(USE_RPC ON)' >> config.cmake
RUN echo 'set(USE_SORT ON)' >> config.cmake
RUN echo 'set(USE_GRAPH_RUNTIME ON)' >> config.cmake
RUN echo 'set(USE_BLAS openblas)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD 14)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD_REQUIRED ON)' >> config.cmake
RUN echo 'set(CMAKE_CXX_EXTENSIONS OFF)' >> config.cmake
#RUN echo 'set(CMAKE_BUILD_TYPE Debug)' >> config.cmake
RUN bash -c \
     "mkdir -p build && \
     cd build && \
     cmake .. && \
     make -j2"
ENV PYTHONPATH=/root/tvm/python:/root/tvm/topi/python:${PYTHONPATH}

# Set up Python
RUN pip3 install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

## Set up example script
WORKDIR /root
COPY ./datagen.py ./datagen.py
COPY ./time-with-tvm.py ./time-with-tvm.py
COPY ./run.sh ./run.sh

CMD ["bash", "run.sh"]
