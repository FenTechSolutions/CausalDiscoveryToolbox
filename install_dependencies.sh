#!/bin/bash


apt-get -q install r-base -y --allow-unauthenticated >/dev/null
apt-get -q install libgmp3-dev  -y --allow-unauthenticated
apt-get -q install build-essential  -y --allow-unauthenticated
apt-get -q install libv8-3.14-dev  -y --allow-unauthenticated
Rscript -e 'install.packages(c("V8","sfsmisc","clue","randomForest","lattice","devtools","MASS"),repos="http://cran.us.r-project.org")' >/dev/null
Rscript -e 'source("http://bioconductor.org/biocLite.R"); biocLite(c("CAM", "SID", "bnlearn", "pcalg", "kpcalg", "D2C"))' >/dev/null
Rscript -e 'library(devtools); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT")' >/dev/null
