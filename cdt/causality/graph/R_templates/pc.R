# R script to prepare and run the PC algorithms

library('pcalg')
library('kpcalg')
library('methods')

runPC <- function(X, suffStat, parentsOf, alpha, variableSelMat, setOptions,
                  directed, verbose,fixedEdges,fixedGaps,
                  result, ...){

  dots <- list(...)
  if(length(dots) > 0){
    warning("options provided via '...' not taken")
  }

  # additional options for PC
  optionsList <- list("indepTest"={CITEST}, "fixedEdges"=fixedEdges,
                      "NAdelete"=TRUE, "m.max"=Inf, "u2pd" = "relaxed",
                      "skel.method"= "stable.fast", "conservative"=FALSE,
                      "maj.rule"=TRUE, "solve.confl"=FALSE, numCores={NJOBS})


  if(is.null(suffStat)){
    suffStat <- list({METHOD_INDEP})  # data=X, ic.method=method_indep)
  }

  pc.fit <- pcalg::pc(suffStat, indepTest = optionsList$indepTest, p=ncol(X),
                      alpha = alpha,
                      fixedGaps= fixedGaps,
                      fixedEdges = fixedEdges ,
                      NAdelete= optionsList$NAdelete, m.max= optionsList$m.max,
                      u2pd=optionsList$u2pd, skel.method= optionsList$skel.method,
                      conservative= optionsList$conservative,
                      maj.rule= optionsList$maj.rule,
                      solve.confl = optionsList$solve.confl,
                      verbose= verbose)

  pcmat <- as(pc.fit@graph, "matrix")

  result <- vector("list", length = length(parentsOf))

  for (k in 1:length(parentsOf)){
    result[[k]] <- which(as.logical(pcmat[, parentsOf[k]]))
    attr(result[[k]],"parentsOf") <- parentsOf[k]
  }

  if(length(parentsOf) < ncol(X)){
    pcmat <- pcmat[,parentsOf]
  }

  list(resList = result, resMat = pcmat)
}

 dataset <- read.csv(file='{FILE}', sep=",");

variableSelMat = {SELMAT}  # NULL
directed = {DIRECTED}  # TRUE
verbose = {VERBOSE} # FALSE
setOptions = {SETOPTIONS} # NULL
parentsOf = 1:ncol(dataset)
alpha <- {ALPHA} # 0.01
if ({SKELETON}){
  fixedGaps <- read.csv(file='{GAPS}', sep=",", header=FALSE) # NULL
  fixedEdges <- read.csv(file='{EDGES}', sep=",", header=FALSE) # NULL
  fixedGaps = (data.matrix(fixedGaps))
  fixedEdges = (data.matrix(fixedEdges))
  rownames(fixedGaps) <- colnames(fixedGaps)
  rownames(fixedEdges) <- colnames(fixedEdges)

}else{
  fixedGaps = NULL
  fixedEdges = NULL
}
result <- runPC(dataset, suffStat = NULL, parentsOf, alpha,
               variableSelMat, setOptions,
               directed, verbose, fixedEdges, fixedGaps, CI_test)

write.csv(result$resMat,row.names = FALSE, file = '{OUTPUT}');
