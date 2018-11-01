ARG REGISTRY=summaplatform
ARG MODEL=deeptag

FROM ${REGISTRY}/models:${MODEL} as model


# FROM kaixhin/cuda-theano:7.5
# FROM nvidia/cuda:8.0-cudnn5-devel

# GPU (CUDA, use with nvidia-docker for GPU support)
# FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
# CPU only
FROM ubuntu:16.04

MAINTAINER Didzis Gosko <didzis.gosko@leta.lv>

RUN apt update
RUN apt install -y python3-pip python3-h5py python3-sklearn curl
RUN pip3 install --upgrade pip six
RUN pip3 install nltk keras==2.0.8 theano==0.9.0 aio-pika==0.21.0

# https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#docker-installation
# RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl

RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt

# Better to leave them in place as long as we are in development mode - UG
# RUN rm -rf /var/lib/apt/lists/*

COPY --from=model /data/word_vectors /opt/app/data/word_vectors
COPY --from=model /data/models /opt/app/data/models
COPY --from=model /data/documents /opt/app/data/test_documents

COPY keras.json /root/.keras/

WORKDIR /opt/app

COPY classify.py /opt/app/
COPY processing.py /opt/app/
COPY deeptagger.py /opt/app/
COPY task.py /opt/app/
COPY rabbitmq.py /opt/app/
COPY worker_pool.py /opt/app/

ENV LANG C.UTF-8
# ENV PYTHONUNBUFFERED y

ENTRYPOINT ["/opt/app/rabbitmq.py"]
