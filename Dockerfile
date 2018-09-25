FROM divkal/cdt-env-base:1.0
MAINTAINER Diviyan Kalainathan <diviyan@lri.fr>
LABEL version="0.2.5"
LABEL description="Docker final image for the Causal Discovery Toolbox"
ARG python

RUN mkdir -p /CDT
COPY . /CDT
RUN cd /CDT && \
    apt-get -q install "python${python}" "python${python}-dev" python3-pip python3-setuptools -y && \
    pip3 install -r requirements.txt && \
    pip3 install pytest pytest-cov && \
    pip3 install codecov && \
    pip3 install .
CMD /bin/sh
