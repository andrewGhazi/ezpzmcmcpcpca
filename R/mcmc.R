#' EZPZ MCMC PCPCA
#' @inheritParams pcpca_mle
#' @param init initialization list or function
#' @param mle_est description
#' @details
#'
#' If set, \code{init} is passed to cmdstanr::sample() as is.
#'
#' For the sake of identifiability, the largest element of the first axis is
#' constrained to be positive with an InvGamma(1,1) distribution, and the cross
#' product of the first two contrastive axes is positive for d = 2.
#' @export
pcpca_mcmc = function(X, Y, gamma, d = 1, verbose = TRUE,
                      mle_est,
                      init = NULL, ...){

  if (is.null(init)) {
    message("Computing MLE estimate to use as initialization point...")

    mle_est = pcpca_mle(X,Y,gamma, d, verbose)
    W_unit = mle_est$W_mle / sqrt(sum(mle_est$W_mle ^2)) * sign(mle_est$W_mle[which.max(abs(mle_est$W_mle))])

    init = function() {

      list(W = W_unit,
           sigma2 = 1)
    }
  }

  data_list = list(n = nrow(X),
                   m = nrow(Y),
                   p = ncol(Y),
                   mle_max_i = mle_est$W_mle |> apply(2, \(.x) which.max(abs(.x))),
                   X = X,
                   k = d,
                   Y = Y,
                   gamma = .85, # what they used in the paper
                   w = 1)

  if (d == 1) {
    model_path = system.file('stan', 'pcpca_gibbs_1d.stan',
                             package = 'ezpzmcmcpcpca')
    data_list$k = NULL
  } else {
    model_path = system.file('stan', 'pcpca_gibbs_Nd.stan',
                             package = 'ezpzmcmcpcpca')
  }

  id_model2 = cmdstanr::cmdstan_model(stan_file = model_path)

  pcpca_fit = id_model2$sample(data = data_list,
                               chains = 4,
                               parallel_chains = 4,
                               init = init, ...)
  pcpca_fit
}
