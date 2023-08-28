functions {

  matrix woodbury(matrix W, real sigma2, int p, int k) {
    // U = W, V = W'
    // https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    real sigma2_inv = 1 / sigma2;

    matrix[p,p] A_inv = sigma2_inv * identity_matrix(p);


    // matrix[p,p] res = A_inv - ((A_inv * W) * inverse(identity_matrix(k) + sigma2_inv * crossprod(W)) * (W' * A_inv));
    // V The inverse_spd() here is kxk, so very fast.
    matrix[p,p] res = A_inv - quad_form(inverse_spd(identity_matrix(k) + sigma2_inv * crossprod(W)), (sigma2_inv * W'));

    return res;
  }

  real log_mdl(real sigma2, int p, int k, matrix W) {
    // https://en.wikipedia.org/wiki/Matrix_determinant_lemma#Generalization
    // not sure if crossprod() is faster than plain matmul
    real res = log_determinant_spd(identity_matrix(k) + 1/sigma2 * crossprod(W)) + p * log(sigma2);

    return res;
  }

  real mult_trace(int p, matrix tA, matrix C) {
    // Don't do the full A * C to get the trace, only do the necessary multiplications.
    // Both A and C are symmetric, so it might be faster to figure out the right factors and use trace_quad_form().
    real res = sum(columns_dot_product(tA, C));

    return res;
  }
}

data {
  int<lower=0> n;          // number of foreground samples
  int<lower=0> m;          // number of background samples
  int<lower=0> p;          // number of features
  int<lower=2, upper=p-1> k;          // latent dim
  array[k] int<lower=0, upper = p> mle_max_i; // index of feature with the largest MLE loading
  matrix[p, k] W_mle;
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
  matrix[p, k] W_raw;
  real<lower=0> sigma2;
  vector<lower=0>[k] W_scales;
}
transformed parameters {
  matrix[p, k] W;
  matrix[p, k] W_ortho;

  # Q factor of std normal matrix = Haar distribution
  # https://nhigham.com/2020/04/22/what-is-a-random-orthogonal-matrix/
  W_ortho = qr_thin_Q(W_raw);
  # Theoretically this can be done 1.5x (O(4/3p^3) instead of O(2p^3)) faster with Householder rotations
  # instead of QR decomposition.

  W = diag_post_multiply(W_ortho, W_scales);
}
model {

  // for (i in 1:k){
  //   target += std_normal_lpdf(W_raw[,k]);
  // } // I don't know why, but this does nothing to constrain W_raw.

  for (i in 1:p) {
    target += std_normal_lpdf(W_raw[i]);
  }

  target += exponential_lpdf(W_scales | 1);
  target += inv_gamma_lpdf(sigma2 | 1,1);

  target += w * (-(n - gamma * m) * 0.5 * log_mdl(sigma2, p, k, W) - 0.5 * mult_trace(p, woodbury(W, sigma2, p, k), C));
}

generated quantities {
  matrix[p,k] W_id;

  W_id = W;

  for (i in 1:k){
    if (dot_product(col(W_id, i), col(W_mle, i)) < 0) {
      W_id[,i] = -1 * W_id[,i];
    }
  }
}
