FROM python:3.8.10-buster
RUN apt-get update
RUN apt-get install -y software-properties-common

RUN apt-get install -y \
    python3-pip python3-dev python3-setuptools \
    --no-install-recommends

RUN apt-get update && apt-get install gettext nano vim -y

RUN pip3 install --upgrade pip

WORKDIR /src
COPY . /src


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt