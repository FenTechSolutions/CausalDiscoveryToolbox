FROM divkal/cdt-env-base:1.1
MAINTAINER Diviyan Kalainathan <diviyan@lri.fr>
LABEL version="0.2.5"
LABEL description="Docker final image for the Causal Discovery Toolbox"
ARG python

RUN mkdir -p /CDT
COPY . /CDT
RUN cd /CDT && \
    apt-get update && \
    apt-get -q install "python${python}" "python${python}-dev" python3-pip python3-setuptools -y && \
    rm /usr/bin/python3 && ln -s /usr/bin/python${python} /usr/bin/python3 && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install pytest pytest-cov && \
    python3 -m pip install codecov && \
    python3 setup.py install
CMD /bin/sh
