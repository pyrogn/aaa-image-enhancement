FROM python:3.10-slim

WORKDIR /usr/src/app

# RUN add-apt-repository ppa:openjdk-r/ppa
# openjdk-11-jdk

ARG GIT_REPO_URL=https://github.com/pyrogn/FFTformer
RUN apt-get update && apt-get install -y git gcc python3-dev openjdk-17-jre-headless libgl1 \
    && git clone ${GIT_REPO_URL} --depth 1 /usr/src/app \
    && apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -r requirements.txt
RUN python setup.py develop --no_cuda_ext

RUN pip install torchserve torch-model-archiver

# I ran it from current folder
COPY . .

RUN torch-model-archiver --model-name deblur \
    --version 1.0 \
    --model-file basicsr/models/archs/fftformer_only.py \
    --serialized-file weights/net_g_Realblur_J.pth \
    --handler fftformer_handler.py 

RUN mkdir /usr/src/app/model_store
RUN mv deblur.mar /usr/src/app/model_store

EXPOSE 8080

CMD ["torchserve", "--start", "--ncs", "--model-store", "model_store", "--models", "my_model=deblur.mar", "--foreground", "--ts-config", "config.properties"]
