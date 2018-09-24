FROM cdt-env-py3.6:1.1
MAINTAINER Diviyan Kalainathan <diviyan@lri.fr>
LABEL version="0.2.5"
LABEL description="Docker dinal image for the Causal Discovery Toolbox"

RUN mkdir -p ~/CDT
COPY . /CDT
RUN cd ~/CDT && pip3 install .
CMD /bin/sh
