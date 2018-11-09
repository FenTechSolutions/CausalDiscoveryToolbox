library(sparsebn)
library(MASS)

dataset <- read.csv(file='{FOLDER}{FILE}', sep=",");
dat <- sparsebnData(as.matrix(dataset), type = "c");
estDAG <- estimate.dag(data = dat);

lambda = select.parameter(estDAG,dat);
write.matrix(get.adjacency.matrix(estDAG[[lambda]]),
             file = '{FOLDER}{OUTPUT}', sep=",");
