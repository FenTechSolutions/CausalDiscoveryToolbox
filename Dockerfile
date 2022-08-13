FROM fentechai/cdt-base:latest
MAINTAINER Diviyan Kalainathan <diviyan@lri.fr>
LABEL description="Docker image for the Causal Discovery Toolbox"
ARG python
ARG spy

RUN mkdir -p /CDT
COPY . /CDT
RUN cd /CDT && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update --allow-unauthenticated && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install pytest pytest-cov && \
    python3 -m pip install codecov && \
    python3 setup.py install
CMD /bin/sh
