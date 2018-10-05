library(methods)
library(bnlearn)
dataset <- read.csv(file='{FOLDER}{FILE}', sep=",");

if({SKELETON}){
  whitelist <- read.csv(file='{FOLDER}{WHITELIST}', sep=",") # NULL
  blacklist <- read.csv(file='{FOLDER}{BLACKLIST}', sep=",") # NULL
}else{
  whitelist = NULL
  blacklist = NULL
}
result <- {ALGORITHM}(dataset, whitelist=whitelist, blacklist=blacklist, alpha={ALPHA}, B={BETA}, optimized={OPTIM} ,test={SCORE}, debug={VERBOSE});
write.csv(result$arc, row.names=FALSE, file = '{FOLDER}{OUTPUT}');
