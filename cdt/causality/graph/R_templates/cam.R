library(CAM)

dataset <- read.csv(file='{FILE}', sep=",");
estDAG <- CAM(dataset, scoreName = "{SCORE}", numCores = {NJOBS}, output = {VERBOSE},
              variableSel = {VARSEL}, variableSelMethod = {SELMETHOD}, pruning = {PRUNING},
              pruneMethod = {PRUNMETHOD}, pruneMethodPars = list(cutOffPVal = {CUTOFF}))
write.csv(as.matrix(estDAG$Adj),row.names = FALSE, file = '{OUTPUT}');
