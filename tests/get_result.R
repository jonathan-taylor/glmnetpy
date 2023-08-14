count <- 144
get_result <- function(i) {
  readRDS(sprintf("test_%03d.RDS", i))
}

#' Deep copy an object, useful for debugging
#' @param object to copy
#' @return a copy of the object
deep_copy <- function(object) {
  return(unserialize(serialize(object, NULL)))
}

i <- 102
d <- get_result(i)
if (d$sparse) {
  d$result <- do.call(glmnet:::spwls_exp, d$args)
} else {
  d$result <- do.call(glmnet:::wls_exp, d$args)
}

tc <- function(x) {
  catn(x)
  identical(m[[x]], d$result[[x]])
}
