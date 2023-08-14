#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
i <- as.integer(args[1])

get_result <- function(i) {
  readRDS(sprintf("test_%03d.RDS", i))
}

#' Deep copy an object, useful for debugging
#' @param object to copy
#' @return a copy of the object
deep_copy <- function(object) {
  return(unserialize(serialize(object, NULL)))
}

## Resave kosher results
d <- get_result(i)
args <- deep_copy(d$args)
if (d$sparse) {
  d$result <- do.call(glmnet:::spwls_exp, args)
} else {
  d$result <- do.call(glmnet:::wls_exp, args)
}
saveRDS(d, sprintf("test_%03d.RDS", i))
pickle(d, fname = sprintf("test_%03d.pickle", i))
