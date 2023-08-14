# TESTS FOR ELNET.FIT (REGULAR AND SPARSE X MATRICES)

# Test for elnet.fit (regular x matrix)
#
# All tests have given weights summing to 1 as elnet.fit does not standardize
# weights. This is necessary in order for the IRLS routine to work properly.
#
# All tests are performed for 2 alpha values and for intercept = TRUE and FALSE.
# Also, all tests have non-equal observation weights and non-equal penalty factors.
#
# - Default test
# - Upper & lower limits
# - Infinite penalty factors
# - Exclude
# - Exclude & Inf penalty factors
# - Exclude & zero penalty factors

# Test for elnet.fit (sparse x matrix)
#
# Same tests as those for elnet.fit. One additional test where xm and xs are
# changed.
library(testthat)
library(glmnet)  ## This is an instrumented version of glmnet to save args for testing!
                 ## See glmnetFlex.R included here. 


# get glmnet "gold standard" for unstandardized data: gaussian family
get_glmnet_val <- function(idx, ...) {
    oldfit <- glmnet(type.gaussian = "naive", standardize = FALSE, ...)
    lambda_seq <- oldfit$lambda[idx]
    oldfit2 <- glmnet(type.gaussian = "naive", standardize = FALSE,
                      lambda = lambda_seq, ...)
    list(fit = oldfit2, lambda_seq = lambda_seq)
}

count <- 0
compare_fits <- function(oldfit, idx, newfit, sparse = FALSE) {
  count <<- count + 1
  ## oldfit$call <- NULL
  ## attr(oldfit, "class") <- NULL
  ## newfit$call <- NULL
  ## attr(newfit, "class") <- NULL
  args <- newfit$args
  out <- list(args = args, sparse = sparse)
  pickle(out, sprintf("test_%03d.pickle", count))
  saveRDS(out, sprintf("test_%03d.RDS", count))
  
  ## expect_equal(as.vector(coef(oldfit)[, idx]), as.vector(coef(newfit)),
  ##              label = paste0("elnet.fit lambda no ", idx, ": coef doesn't match"))
  ## expect_equal(oldfit$df[idx], newfit$df,
  ##              label = paste0("elnet.fit lambda no ", idx, ": df doesn't match"))
  ## expect_equal(oldfit$lambda[idx], newfit$lambda,
  ##              label = paste0("elnet.fit lambda no ", idx, ": lambda doesn't match"))
  ## expect_equal(oldfit$dev.ratio[idx], newfit$dev.ratio,
  ##              label = paste0("elnet.fit lambda no ", idx, ": dev.ratio doesn't match"))
  ## expect_equal(oldfit$nulldev, newfit$nulldev,
  ##              label = paste0("elnet.fit lambda no ", idx, ": nulldev doesn't match"))
  ## expect_equal(oldfit$nobs, newfit$nobs,
  ##              label = paste0("elnet.fit lambda no ", idx, ": nobs doesn't match"))
  
}

# wrapper for running the elnet.fit() tests
# params: variable args to be passed to elnet.fit (except intercept
# and alpha, which are provided by run_test)
# sparse: is x matrix sparse? only used for naming test
run_test <- function(test_name, alpha_list, lambda_idx, params, sparse = FALSE) {
    for (alpha in alpha_list) {
        for (intercept in c(1, 0)) {
            test_that(paste("elnet.fit", sparse, test_name, ": alpha", alpha, "intr", intercept), {
                params[["intercept"]] <- intercept
                params[["alpha"]] <- alpha
                target <- do.call(get_glmnet_val, append(params, list(idx = lambda_idx)))

                # first lambda
                wls_params <- list(lambda = target$lambda_seq[1], save.fit = TRUE)
                wls_fit1 <- do.call(glmnet:::elnet.fit, append(params, wls_params))
                compare_fits(target$fit, 1, wls_fit1, sparse = sparse)

                # second lambda w warmstart
                wls_params <- list(lambda = target$lambda_seq[2], warm = wls_fit1)
                wls_fit2 <- do.call(glmnet:::elnet.fit, append(params, wls_params))
                compare_fits(target$fit, 2, wls_fit2, sparse = sparse)
                
                # second lambda w warmstart (just coefs)
                wls_params[["warm"]] <- list(beta = wls_fit1$warm_fit$a, a0 = wls_fit1$warm_fit$aint)
                wls_fit3 <- do.call(glmnet:::elnet.fit, append(params, wls_params))
                compare_fits(target$fit, 2, wls_fit3, sparse = sparse)
            })
        }
    }
}

#####
# ELNET.FIT TESTS
#####
context("elnet.fit()")

# parameters for data (and weights for glmnet & data)
nobs <- 200; nvars <- 20
beta <- matrix(c(2, 2, 4, -5, rep(0, nvars - 4)), ncol = 1)
weights <- rep(1:3, length.out = nobs)
# elnet.fit does NOT standardize observation weights while glmnet does.
# adding line below to ensure compatibility btw the two
weights <- weights / sum(weights)

# set up fake data (rescale y in the same way that glmnet would)
set.seed(50)
x <- matrix(rnorm(nobs * nvars), nrow = nobs)
y <- x %*% beta + rnorm(nobs)
my <- weighted.mean(y,weights)
sy <- sqrt(weighted.mean((y-my)^2,weights))
y <- scale(y, my, sy)

# fitting parameters common to all tests
base_params <- list(x = x, y = y, weights = weights)
vp <- rep(1:2, length.out = nvars)  # used often

# test list
test_param_list <- list(
    "basic" = list(penalty.factor = vp, thresh = 1e-12),
    "UL limits" = list(penalty.factor = vp, thresh = 1e-12, upper.limits = 0.2, 
                       lower.limits = -0.2),
    "Inf vp" = list(penalty.factor = c(Inf, Inf, rep(1:2, length.out = nvars - 2)),
                    thresh = 1e-12),
    "exclude" = list(penalty.factor = vp, thresh = 1e-12, exclude = 1:2),
    "exclude & Inf vp" = list(penalty.factor = c(1, Inf, 1, 2, Inf, 
                                                 rep(2:1, length.out = nvars - 5)),
                              exclude = 1:2, thresh = 1e-12),
    "exclude & 0 vp" = list(penalty.factor = c(1, 2, 0, 0, 
                                               rep(1:2, length.out = nvars - 4)),
                            exclude = c(1, 3), thresh = 1e-14)
)

# run tests
for (test_name in names(test_param_list)) {
    run_test(test_name, alpha_list = c(1.0, 0.3), lambda_idx = 40:41, 
             append(base_params, test_param_list[[test_name]]))
}

#####
# ELNET.FIT TESTS (SPARSE X MATRIX)
#####
context("elnet.fit(), sparse x")

# parameters for data (and weights for glmnet & data)
nobs <- 200; nvars <- 30
beta <- matrix(c(2, 2, 4, -5, -3, rep(0, nvars - 5)), ncol = 1)
weights <- rep(1:3, length.out = nobs)
# elnet.fit does NOT standardize observation weights while glmnet does.
# adding line below to ensure compatibility btw the two
weights <- weights / sum(weights)

# set up fake data (rescale y in the same way that glmnet would)
set.seed(50)
xvec <- rnorm(nobs * nvars)
xvec[sample.int(nobs * nvars, size = 0.6 * nobs * nvars)] <- 0
x <- Matrix::Matrix(xvec, nrow = nobs, sparse = TRUE)
attr(x, "xm") <- rep(0, times = nvars)
attr(x, "xs") <- rep(1, times = nvars)
y <- x %*% beta + 2 * rnorm(nobs)
my <- weighted.mean(y,weights)
sy <- sqrt(weighted.mean((y-my)^2,weights))
y <- scale(y, my, sy)

# fitting parameters common to all tests
base_params <- list(x = x, y = y, weights = weights)
vp <- rep(1:2, length.out = nvars)  # used often

# test list
test_param_list <- list(
    "basic" = list(penalty.factor = vp, thresh = 1e-12),
    "UL limits" = list(penalty.factor = vp, thresh = 1e-15, 
                       upper.limits = 0.2, lower.limits = -0.2),
    "Inf vp" = list(thresh = 1e-10,
                    penalty.factor = c(Inf, Inf, rep(1:2, length.out = nvars - 2))),
    "exclude" = list(penalty.factor = vp, thresh = 1e-14,
                     exclude = 1:2),
    "exclude & Inf vp" = list(penalty.factor = c(1, Inf, 1, 2, Inf, 
                                                 rep(2:1, length.out = nvars - 5)),
                              exclude = 1:2, thresh = 1e-12),
    "exclude & 0 vp" = list(penalty.factor = c(1, 2, 0, 0, 
                                               rep(1:2, length.out = nvars - 4)),
                            exclude = c(1, 3), thresh = 1e-14)
)

# run tests
for (test_name in names(test_param_list)) {
    run_test(test_name, alpha_list = c(1.0, 0.3), lambda_idx = 20:21, 
              append(base_params, test_param_list[[test_name]]), sparse = TRUE)
}

## #####
## # basic test with change in xm and xs: unfortunately need special code for this
## #####
## thresh <- 1e-14
## lambda_idx <- 20:21
## attr(x, "xm") <- rep(0.1, times = nvars)
## attr(x, "xs") <- rep(2, times = nvars)
## for (alpha in c(1.0)) {
##     for (intercept in c(1, 0)) {
##         test_that(paste("alpha", alpha, "intercept", intercept), {
##             target <- get_glmnet_val(lambda_idx, x = (x-0.1)/2, y = y, 
##                                      intercept = intercept, alpha = alpha, 
##                                      weights = weights, penalty.factor = vp, 
##                                      thresh = thresh)

##             # first lambda
##             wls_fit1 <- glmnet:::elnet.fit(x, y, weights, target$lambda_seq[1], 
##                                            alpha, intercept,
##                                            penalty.factor = vp, thresh = thresh,
##                                            save.fit = TRUE)
##             compare_fits(target$fit, 1, wls_fit1)

##             # second lambda w warmstart
##             wls_fit2 <- glmnet:::elnet.fit(x, y, weights, target$lambda_seq[2], 
##                                            alpha, intercept,
##                                            penalty.factor = vp, thresh = thresh,
##                                            warm = wls_fit1)
##             compare_fits(target$fit, 2, wls_fit2)
##         })
##     }
## }

