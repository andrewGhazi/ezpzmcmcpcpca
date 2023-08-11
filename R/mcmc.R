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

      list(W_scales = apply(mle_est$W_mle, 2, \(x) sqrt(sum(x^2))),
           sigma2 = 1,
           W_raw = W_unit)
    }
  }

  data_list = list(n = nrow(X),
                   m = nrow(Y),
                   p = ncol(Y),
                   mle_max_i = mle_est$W_mle |> apply(2, \(.x) which.max(abs(.x))),
                   # W_signs = sign(mle_est$W_mle),
                   W_mle = mle_est$W_mle,
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
    model_path = system.file('stan', 'pcpca_gibbs_Nd_ortho.stan',
                             package = 'ezpzmcmcpcpca')
  }

  id_model2 = cmdstanr::cmdstan_model(stan_file = model_path)

  pcpca_fit = id_model2$sample(data = data_list,
                               init = init, ...)

  pcpca_fit
  # stop("Ghazi needs to finish get_W_id()!"); pcpca_fit |> get_W_id(p = ncol(Y), k = d)
}

# These two functions dot the draws of W onto the posterior mean of the first
# chain (instead of W_mle like the sampler) to get W_id. It doesn't really make
# a difference.
dot_to_11 = function(draw, draw11, p, k) {

  # For each latent dimension in the draw, check if it points in roughly the
  # same direction as draw #1, according to the sign of the dot product. If not,
  # multiply it by -1.

  dot_prods = draw %*% d11w

  for (i in 1:k) {
    if (dot_prods[1,i] < 0) {
      i_cols = (1:p)+ (i-1)*p
      draw[i_cols] = -1*draw[i_cols]
    }
  }

  return(draw)
}

get_W_id = function(pcpca_fit, p, k) {
  draw_mat = pcpca_fit$draws("W", format = 'matrix')

  draw11 = pcpca_fit$draws('W')[,1,] |>
    posterior::summarise_draws(mean = mean) |>
    getElement("mean")

  d11w = matrix(nrow = p*k, ncol = k, 0)
  for (i in 1:k){
    d11w[(1:p) + (i-1)*p,i] = draw11[(1:p) + (i-1)*p]
  }

  W_id = draw_mat |> apply(1, dot_to_11, d11w, p, k) |> t()

  colnames(W_id) = colnames(W_id) |>
    gsub("W", "W_id", x = _)

  return(W_id)

}
