FROM python:3.10-slim

WORKDIR /usr/src/app

# RUN add-apt-repository ppa:openjdk-r/ppa
# openjdk-11-jdk

ARG GIT_REPO_URL=https://github.com/pyrogn/Illumination-Adaptive-Transformer.git
RUN apt-get update && apt-get install -y git gcc python3-dev openjdk-17-jre-headless libgl1 \
    && git clone ${GIT_REPO_URL} --depth 1 /usr/src/app \
    && apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -r requirements.txt

RUN pip install -e .

RUN pip install torchserve torch-model-archiver

WORKDIR /usr/src/app/IAT_enhance


# ещё не знаю, какую модель использовать
RUN torch-model-archiver --model-name lowlight \
    --version 1.0 \
    --model-file model/IAT.py \
    --serialized-file best_Epoch_exposure.pth \ 
    --handler model-handler.py 

RUN mkdir /usr/src/app/model_store
RUN mv lowlight.mar /usr/src/app/model_store

WORKDIR /usr/src/app

EXPOSE 8080

# I ran it from current folder
COPY . .

CMD ["torchserve", "--start", "--ncs", "--model-store", "model_store", "--models", "my_model=lowlight.mar", "--foreground", "--ts-config", "config.properties"]
