
<!-- README.md is generated from README.Rmd. Please edit that file -->

# ezpzmcmcpcpca

R implementation of PCPCA. PCPCA is the work of Didong Li, Andrew Jones,
and Barbara Engelhardt. Original manuscript here:
<https://arxiv.org/abs/2012.07977> and original Python implementation
here: <https://github.com/andrewcharlesjones/pcpca>

The MCMC implementation here is derived from the original Stan code but
uses a few linear algebra identities that make it comparatively faster
(Woodbury matrix inverse, matrix determinant lemma, fast multiply+trace)
and more interpretable (Haar distribution on contrastive axes to ensure
they’re orthogonal).

<!-- badges: start -->
<!-- badges: end -->

## Installation

This package depends on CmdStan & cmdstanr, which are not available via
the usual CRAN route.

``` r

install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))

library(cmdstanr)

check_cmdstan_toolchain()

install_cmdstan(cores = 2)

remotes::install_github("andrewGhazi/ezpzmcmcpcpca")
```

If the `cmdstanr` installation doesn’t work you can find more detailed
instructions [at this link](https://mc-stan.org/cmdstanr/).

## Example

The package has two functions `pcpca_mle()` and `pcpca_mcmc()`. The
former directly implements the MLE estimates from the paper. The second
adapts the Gibbs posterior provided in the original python
implementation.

``` r
library(ezpzmcmcpcpca)

set.seed(123)
mu_vec = c(-.75, .75)
Sigma = matrix(nrow = 2, c(1, .7, .7, 1))

X1 = MASS::mvrnorm(100, 
                   mu = mu_vec,
                   Sigma = Sigma)

X2 = MASS::mvrnorm(100, 
                   mu = -1 * mu_vec,
                   Sigma = Sigma)

Y = MASS::mvrnorm(200, 
                  mu = rep(0,2),
                  Sigma = Sigma)
X = rbind(X1, X2)

plot(rbind(Y, X1, X2),
     pch = 19, 
     col = c(rep('grey', 200), 
             rep('dodgerblue2', 100), 
             rep('firebrick1', 100)),
     xlab = 'dim_1', ylab = 'dim_2',
     main = "Contrastive axis on 2D data")

mle_estimate = pcpca_mle(X,Y, gamma = .85)
#> The first 2 of 2 eigenvalues of C are positive.

# $sigma2_mle
# [1] 0.01181884
# 
# $W_mle
#            [,1]
# [1,] -0.1142008
# [2,]  0.1195985

mcmc_estimate = pcpca_mcmc(X, Y, .85, d = 1,
                           mle_est = mle_estimate, 
                           refresh = 0)
#> Computing MLE estimate to use as initialization point...
#> Running MCMC with 4 sequential chains...
#> 
#> Chain 1 finished in 0.0 seconds.
#> Chain 2 finished in 0.0 seconds.
#> Chain 3 finished in 0.0 seconds.
#> Chain 4 finished in 0.0 seconds.
#> 
#> All 4 chains finished successfully.
#> Mean chain execution time: 0.0 seconds.
#> Total execution time: 0.6 seconds.

W_draws = mcmc_estimate$draws('W', format = 'matrix')[sample.int(4000,40),]

# > mcmc_estimate
#   variable   mean median   sd  mad     q5    q95 rhat ess_bulk ess_tail
#  lp__      -42.85 -42.51 1.31 1.07 -45.41 -41.43 1.00     1269     1860
#  W[1,1]      2.15   2.12 0.30 0.29   1.71   2.70 1.00     1406     1312
#  W[2,1]     -1.98  -1.95 0.29 0.27  -2.50  -1.57 1.00     1384     1329
#  sigma2      0.24   0.23 0.07 0.07   0.15   0.38 1.00     2024     1901
#  ...

arrows(x0 = 0, y0 = 0,
       x1 = mle_estimate$W_mle[1,1],
       y1 = mle_estimate$W_mle[2,1], 
       col = "limegreen", length = .075, lwd = 3)
arrows(x0 = 0, y0 = 0,
       x1 = W_draws[,1],
       y1 = W_draws[,2], 
       col = rgb(0,0,0,.333), length = .1)
```

<img src="man/figures/README-example-1.png" width="100%" />

# Identifiability

It also works for higher numbers of latent dimensions. The simulation
below explicitly defines two contrastive axes (`W_rand`), but one could
also simply create multiple groups in X with varying mean.

A note on identifiability. The sign of the contrastive axes are not
identifiable. Axes pointing like this `┘` have the same posterior
density as axes flipped to point like this `┌`.

To help get around this, we produce a generated quantity variable `W_id`
that dots each axis onto the MLE, and if it’s negative multiplies by -1.
This works well if there is sufficient data, and the convergence metrics
of `W_id` (particularly Rhat) will be happy. Higher numbers of latent
dimensions requires more data for clear inference.

Nonetheless, this doesn’t always work. If you notice component(s) of
`W_id` that seem to be symmetrically flipping about zero, that means
there’s substantial probability that the posterior mean isn’t pointing
in the right direction. This diagnostic use is one of the main
advantages of the MCMC approach.

``` r
library(ezpzmcmcpcpca)

set.seed(123)
p = 10
k = 2

# more variance along the second axis
W_rand = matrix(rnorm(p*k), ncol = k) %*% diag(c(2,.75)); W_rand
#>             [,1]        [,2]
#>  [1,] -1.1209513  0.91806135
#>  [2,] -0.4603550  0.26986037
#>  [3,]  3.1174166  0.30057859
#>  [4,]  0.1410168  0.08301204
#>  [5,]  0.2585755 -0.41688085
#>  [6,]  3.4301300  1.34018485
#>  [7,]  0.9218324  0.37338786
#>  [8,] -2.5301225 -1.47496287
#>  [9,] -1.3737057  0.52601693
#> [10,] -0.8913239 -0.35459356

cor_mat = trialr::rlkjcorr(1, p, 1);

Y = MASS::mvrnorm(n = 500, mu = rep(0,p),
                  Sigma = cor_mat)

contrastive_noise = MASS::mvrnorm(n = 500,
                                  Sigma = diag(k),
                                  mu = rep(0,k))

X = MASS::mvrnorm(n = 500, Sigma = cor_mat,
                  mu = rep(0,p)) +
  (contrastive_noise %*% t(W_rand))


res_mle = pcpca_mle(X, Y, .85, 2); res_mle
#> The first 8 of 10 eigenvalues of C are positive.
#> $sigma2_mle
#> [1] 0.002039828
#> 
#> $W_mle
#>              [,1]          [,2]
#>  [1,] -0.09051975  0.1441503495
#>  [2,] -0.04124558  0.0224530175
#>  [3,]  0.35602831 -0.0736810563
#>  [4,]  0.02689480 -0.0151457993
#>  [5,]  0.02015256 -0.0459358586
#>  [6,]  0.42501649  0.0554328880
#>  [7,]  0.11986501  0.0002425385
#>  [8,] -0.32642680 -0.0951041155
#>  [9,] -0.13221252  0.1036616262
#> [10,] -0.11940364 -0.0051256520
res_mcmc = pcpca_mcmc(X, Y, .85, 2, 
                      mle_est = res_mle,
                      chains = 4,
                      parallel_chains = 4, refresh = 0,
                      show_messages = FALSE)
#> Computing MLE estimate to use as initialization point...
#> Running MCMC with 4 parallel chains...
#> 
#> Chain 1 finished in 1.9 seconds.
#> Chain 3 finished in 1.9 seconds.
#> Chain 2 finished in 2.0 seconds.
#> Chain 4 finished in 2.0 seconds.
#> 
#> All 4 chains finished successfully.
#> Mean chain execution time: 2.0 seconds.
#> Total execution time: 2.1 seconds.
#> Warning: 1 of 4000 (0.0%) transitions ended with a divergence.
#> See https://mc-stan.org/misc/warnings for details.

res_mcmc$summary(c("sigma2", "W_id"))
#> # A tibble: 21 × 10
#>    variable    mean median     sd    mad     q5    q95  rhat ess_bulk ess_tail
#>    <chr>      <num>  <num>  <num>  <num>  <num>  <num> <num>    <num>    <num>
#>  1 sigma2     1.05   1.05  0.0622 0.0619  0.955  1.16   1.00    3635.    2644.
#>  2 W_id[1,1] -1.88  -1.89  0.444  0.443  -2.62  -1.17   1.00    3524.    2372.
#>  3 W_id[2,1] -0.858 -0.853 0.145  0.141  -1.10  -0.627  1.00    4580.    2899.
#>  4 W_id[3,1]  7.38   7.34  0.599  0.591   6.45   8.41   1.00    4523.    2865.
#>  5 W_id[4,1]  0.560  0.559 0.126  0.128   0.356  0.770  1.00    4937.    3048.
#>  6 W_id[5,1]  0.419  0.420 0.174  0.172   0.132  0.707  1.00    3818.    2416.
#>  7 W_id[6,1]  8.80   8.76  0.675  0.668   7.78   9.98   1.00    4427.    2685.
#>  8 W_id[7,1]  2.49   2.47  0.216  0.212   2.15   2.87   1.00    4510.    2922.
#>  9 W_id[8,1] -6.75  -6.72  0.566  0.558  -7.72  -5.89   1.00    3957.    2973.
#> 10 W_id[9,1] -2.74  -2.73  0.371  0.365  -3.37  -2.14   1.00    3432.    2824.
#> # ℹ 11 more rows

# Look at the first component
res_mcmc$draws('W_id', format = 'matrix')[,1:10] |>
  bayesplot::mcmc_trace() 
```

<img src="man/figures/README-unnamed-chunk-3-1.png" width="100%" />

``` r

contrastive_magnitude = sqrt(rowSums(contrastive_noise^2))
pal_fun = colorRampPalette(c('grey', 'red'))

contrast_cols = pal_fun(20)[cut((contrastive_magnitude / max(contrastive_magnitude)),
                                breaks = 20) |> as.numeric()]

rbind(X) %*% matrix(res_mcmc$summary("W_id")$mean, ncol = 2) |> 
  plot(xlab = "", ylab = "",
       main = "10D data projected to first two contrastive axes",
       sub = "Points colored by magnitude of true contrastive noise",
       col = contrast_cols, 
       pch = 19)
```

<img src="man/figures/README-unnamed-chunk-3-2.png" width="100%" />

``` r

plot(W_rand[1:(p*k)], matrix(res_mcmc$summary("W_id")$mean, ncol = 1), 
     pch = 19, 
     xlab = 'true axis components', ylab = 'posterior means', 
     main = "True vs estimated axis components")
```

<img src="man/figures/README-unnamed-chunk-3-3.png" width="100%" />

# Future goals

- Infer gamma probabilistically (likely requires a strong prior)
- further speedups (lowest hanging fruit at this point is probably to
  avoid forming the full pxp matrix in `woodbury()` by also passing in C
  and only keeping track of the differences)
