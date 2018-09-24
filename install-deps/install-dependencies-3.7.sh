#!/bin/bash

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
apt-get -q install r-base -y --allow-unauthenticated
apt-get -q install libssl-dev -y
apt-get -q install libgit2-dev -y
apt-get -q install git -y
apt-get -q install libgmp3-dev  -y --allow-unauthenticated
apt-get -q install build-essential  -y --allow-unauthenticated
apt-get -q install libv8-3.14-dev  -y --allow-unauthenticated
apt-get -q install libcurl4-openssl-dev -y --allow-unauthenticated
apt-get -q install python3.7 python3.7-dev python3-pip python3-setuptools -y
Rscript -e 'install.packages(c("V8","sfsmisc","clue","randomForest","lattice","devtools","MASS"),repos="http://cran.us.r-project.org")'
Rscript -e 'source("http://bioconductor.org/biocLite.R"); biocLite(c("CAM", "SID", "bnlearn", "pcalg", "kpcalg", "D2C"))'
Rscript -e 'library(devtools); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT")'
pip3 install -r requirements.txt
pip3 install pytest pytest-cov
pip3 install codecov
