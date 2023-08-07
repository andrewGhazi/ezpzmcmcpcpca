functions {
  // https://en.wikipedia.org/wiki/Woodbury_matrix_identity
  matrix woodbury(matrix W, real sigma2, int p, int k) {
    // U = W, V = W'
    real sigma2_inv = 1 / sigma2;

    matrix[p,p] A_inv = sigma2_inv * identity_matrix(p);


    // matrix[p,p] res = A_inv - ((A_inv * W) * inverse(identity_matrix(k) + sigma2_inv * crossprod(W)) * (W' * A_inv));
    // V The inverse_spd() here is kxk, so very fast.
    matrix[p,p] res = A_inv - quad_form(inverse_spd(identity_matrix(k) + sigma2_inv * crossprod(W)), (W' * A_inv));

    return res;
  }

  real log_mdl(real sigma2, int p, int k, matrix W) {
    // https://en.wikipedia.org/wiki/Matrix_determinant_lemma#Generalization
    // not sure if crossprod() is faster than plain matmul
    real res = log_determinant_spd(identity_matrix(k) + 1/sigma2 * crossprod(W)) + p * log(sigma2);

    return res;
  }
}
data {
  int<lower=0> n;          // number of foreground samples
  int<lower=0> m;          // number of background samples
  int<lower=0> p;          // number of features
  int<lower=0, upper=p-1> k;          // latent dim
  array[k] int<lower=0, upper = p> mle_max_i; // index of feature with the largest MLE loading

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
  matrix[p, k] W;
  real<lower=0> sigma2;
}
transformed parameters {
  matrix[p, p] A;
  matrix[p, p] Wtcp = tcrossprod(W);

  A = Wtcp + sigma2 * identity_matrix(p);
}
model {
  //prior on the first element of W being positive to make it identifiable
  // would be better to have a prior on the largest element. Maybe get that from the MLE and pass it in.
  for (i in 1:k) {
    target += inv_gamma_lpdf(W[mle_max_i[i],i] | 1,1);
  }
  // target += std_normal_lpdf(W[,1]);
  // target += inv_gamma_lpdf(sqrt(W[1,1]^2 + W[2,1]^2)| 1,1);
  // target += inv_gamma_lpdf(sigma2 | 2,2);

  target += w * (-(n - gamma * m) * 0.5 * log_mdl(sigma2, p, k, W) - 0.5 * trace(woodbury(W, sigma2, p, k) * C));
}
