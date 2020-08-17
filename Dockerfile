FROM fentechai/cdt-base:20.07
MAINTAINER Diviyan Kalainathan <diviyan@lri.fr>
LABEL description="Docker image for the Causal Discovery Toolbox"
ARG python
ARG spy

RUN mkdir -p /CDT
COPY . /CDT
RUN cd /CDT && \
    apt-get update --allow-unauthenticated && \
    apt-get -q install "python${python}" "python${python}-dev" python3-pip python3-setuptools -y && \
    rm /usr/bin/python3 && ln -s /usr/bin/python${python} /usr/bin/python3 && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install pytest pytest-cov && \
    python3 -m pip install codecov && \
    python3 setup.py install
CMD /bin/sh
