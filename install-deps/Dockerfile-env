ARG nv
FROM ubuntu
MAINTAINER Diviyan Kalainathan <diviyan@lri.fr>
LABEL version="0.2.5"
LABEL description="Docker image for the Causal Discovery Toolbox"

COPY . /tmp
COPY "install-deps/install-dependencies.sh" /tmp
RUN cd /tmp && bash "install-dependencies.sh"
CMD /bin/sh
