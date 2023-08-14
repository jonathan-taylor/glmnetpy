## Number of saved tests from R wls and sp_wls

num_tests <- 144

get_result <- function(i) {
  d <- readRDS(sprintf("test_%03d.RDS", i))
  if (d$sparse) {
    result <- do.call(glmnet:::spwls_exp, d$args)
  } else {
    result <- do.call(glmnet:::wls_exp, d$args)
  }
  d$result <- result
  d
}

#' Deep copy an object, useful for debugging
#' @param object to copy
#' @return a copy of the object
deep_copy <- function(object) {
  return(unserialize(serialize(object, NULL)))
}

d <- get_result(4)
d1 <- deep_copy(d)
d2 <- deep_copy(d)
res <- do.call(glmnet:::wls_exp, d$args)
res1 <- do.call(glmnet:::wls_exp, d1$args)
res2 <- do.call(glmnet:::wls_exp, d2$args)
