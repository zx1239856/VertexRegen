FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip libcgal-dev libeigen3-dev
