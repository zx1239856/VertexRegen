FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ARG DOCKER_USER=default USER_HOME=/workspace

ARG TOOLS="nvtop tmux htop vim"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y sudo git ninja-build $TOOLS $EXTRA_DEPS && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    adduser --disabled-password --gecos '' --home $USER_HOME $DOCKER_USER && adduser $DOCKER_USER sudo && \
    chown -R $DOCKER_USER:$DOCKER_USER $USER_HOME

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX" FORCE_CUDA=1

ADD requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && MAX_JOBS=24 pip install "flash-attn==2.8.3" --no-build-isolation

USER $DOCKER_USER
