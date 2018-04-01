library(methods)
library(pcalg)

dataset <- read.csv(file='{FILE}', sep=",");

if({SKELETON}){
  fixedGaps <- read.csv(file='{GAPS}', sep=",", header=FALSE) # NULL
  fixedGaps = (data.matrix(fixedGaps))
  rownames(fixedGaps) <- colnames(fixedGaps)
}else{
  fixedGaps = NULL
}
score <- new("{SCORE}", data = dataset)
result <- pcalg::gies(score, fixedGaps=fixedGaps)
gesmat <- as(result$essgraph, "matrix")
gesmat[gesmat] <- 1
  #gesmat[!gesmat] <- 0
write.csv(gesmat, row.names=FALSE, file = '{OUTPUT}');
