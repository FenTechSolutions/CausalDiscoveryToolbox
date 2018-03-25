library(SID)
library(methods)

target <- read.csv(file="{target}" , header=FALSE, sep=",")
prediction <- read.csv(file="{prediction}" , header=FALSE, sep=",")

SID_dist <- structIntervDist(target, prediction)$sid
write.table(SID_dist, '{result}', row.names=FALSE, col.names=FALSE)
