functions {
  // These two functions are for k = 1 only.
  // https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
  matrix sherman_morrison2(matrix W, matrix Wtcp, real sigma2, int p) {
    // W must be a one-column matrix

    real sigma2_inv = 1 / sigma2;

    matrix[p,p] A_inv = sigma2_inv * identity_matrix(p);

    matrix[p,p] res = A_inv - sigma2_inv^2 * Wtcp / (1 + (sigma2_inv * dot_self(to_vector(W))));
    return res;
  }

  real log_mdl(real sigma2, int p, matrix W) {
    // https://en.wikipedia.org/wiki/Matrix_determinant_lemma

    real res = log(1 + 1/sigma2 * dot_self(to_vector(W))) + p*log(sigma2);

    return res;
  }
}
data {
  int<lower=0> n;          // number of foreground samples
  int<lower=0> m;          // number of background samples
  int<lower=0> p;          // number of features
  int<lower=0, upper = p> mle_max_i; // index of feature with the largest MLE loading

  matrix[m, p] Y;          // background data
  matrix[n, p] X;          // foreground data
  real<lower=0> gamma;     // PCPCA tuning parameter
  real<lower=0> w;         // learning rate
}
transformed data {
  matrix[p, p] C;
  C = crossprod(X) - gamma * crossprod(Y);
}
parameters {
  matrix[p, 1] W;
  real<lower=0> sigma2;
}
// transformed parameters {
//   // matrix[p, p] A;
//   // matrix[p, p] Wtcp = tcrossprod(W);
//
//   // A = Wtcp + sigma2 * identity_matrix(p);
// }
model {
  //prior on the first element of W being positive to make it identifiable
  // would be better to have a prior on the largest element. Maybe get that from the MLE and pass it in.
  target += inv_gamma_lpdf(W[mle_max_i,1] | 1,1);
  // target += std_normal_lpdf(W[,1]);
  // target += inv_gamma_lpdf(sqrt(W[1,1]^2 + W[2,1]^2)| 1,1);
  // target += inv_gamma_lpdf(sigma2 | 2,2);

  target += w * (-(n - gamma * m) * 0.5 * log_mdl(sigma2, p, W) - 0.5 * trace(sherman_morrison2(W, tcrossprod(W), sigma2, p) * C));
}
