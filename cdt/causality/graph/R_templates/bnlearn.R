library(methods)
library(bnlearn)
dataset <- read.csv(file='{FILE}', sep=",");

if({SKELETON}){
  whitelist <- read.csv(file='{WHITELIST}', sep=",") # NULL
  blacklist <- read.csv(file='{BLACKLIST}', sep=",") # NULL
}else{
  whitelist = NULL
  blacklist = NULL
}
result <- {ALGORITHM}(dataset, whitelist=whitelist, blacklist=blacklist, alpha={ALPHA}, B={BETA}, optimized={OPTIM} ,test={SCORE}, debug={VERBOSE});
write.csv(result$arc, row.names=FALSE, file = '{OUTPUT}');
