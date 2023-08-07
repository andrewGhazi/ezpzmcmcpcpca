#' PCPCA MLE
#' @param X matrix of the FOREGROUND data
#' @param Y matrix of the BACKGROUND data
#' @param gamma PCPCA tuning parameter
#' @param d latent dimension to project to
#' @param verbose verbosity switch
#' @returns a list with two elements: sigma2_mle and W_mle
#' @export
pcpca_mle = function(X, Y, gamma, d = 1,
                     verbose = TRUE) {

  D = ncol(X)

  n = nrow(X)
  m = nrow(Y)

  # These can be expensive if X or Y are big.
  cov_bg = cov(Y)
  cov_fg = cov(X)

  eig_bg = eigen(cov_bg)
  eig_fg = eigen(cov_fg)

  # Condtioning checks here

  C = cov_fg - gamma*cov_bg

  eigC = eigen(C) # C is almost never posdef

  # message about the first d eigenvalues of C:

  n_pos = sum(eigC$values > 0)

  if (verbose) message(paste0("The first ", n_pos, " of ", D, " eigenvalues of C are positive."))

  U_d = eigC$vectors[,1:d, drop = FALSE]

  sigma2_mle = 1 / ((n - gamma*m) * (D-d)) *
    sum(eigC$values[(d+1):D]); sigma2_mle

  lambda_d = diag(d)
  diag(lambda_d) = eigC$values[1:d]

  W_mle = U_d %*% (lambda_d / (n - gamma*m) - (sigma2_mle * diag(d)))^(1/2)

  return(list(sigma2_mle = sigma2_mle,
              W_mle = W_mle))

}
