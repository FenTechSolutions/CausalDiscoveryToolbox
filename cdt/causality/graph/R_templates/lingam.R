library(pcalg)

dataset <- read.csv(file='{FOLDER}{FILE}', sep=",");
estDAG <- lingam(dataset, verbose = {VERBOSE})
write.csv(as.matrix(estDAG$Bpruned),row.names = FALSE, file = '{FOLDER}{OUTPUT}');
