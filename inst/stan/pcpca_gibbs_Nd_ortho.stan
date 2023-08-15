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

  // vector vec_proj(vector v, vector u, int p) {
  //   // projection of v onto u
  //   vector[p] res = (dot_product(u, v) / dot_self(u)) * u;
  //   return res;
  // }
  //
  // vector unitize(vector u) {
  //   return u / sqrt(dot_self(u));
  // }
  //
  // int signnum(real x) {
  //   return x < 0 ? -1 : x > 0;
  // }
}
data {
  int<lower=0> n;          // number of foreground samples
  int<lower=0> m;          // number of background samples
  int<lower=0> p;          // number of features
  int<lower=2, upper=p-1> k;          // latent dim
  array[k] int<lower=0, upper = p> mle_max_i; // index of feature with the largest MLE loading

  // array[p,k] int<lower=-1, upper=1> W_signs;
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
  // matrix[p, p] A;

  // Gram-Schmidt orthogonalization of W_raw
  // https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
  // W_ortho = W_raw;

  // non_ortho_comp = dot_product(col(W,1), col(W,2));

  // for (i in 2:k) {
  //   vector[p] non_ortho_comp;
  //   non_ortho_comp = rep_vector(0, p);
  //   for (j in 1:(i-1)) {
  //     non_ortho_comp = non_ortho_comp + vec_proj(col(W_ortho,i), col(W_ortho,j), p);
  //   }
  //   W_ortho[,i] = unitize(W_raw[,i] - non_ortho_comp);
  // }

  W_ortho = qr_thin_Q(W_raw);

  // TODO: dot columns of W_ortho onto columns of W_mle. If it's negative, flip it.

  // for (i in 1:k){
  //   if (dot_product(col(W_ortho, i), col(W_mle, i)) < 0) {
  //     W_ortho[,i] = -1 * W_ortho[,i];
  //   }
  // }

  W = diag_post_multiply(W_ortho, W_scales);

  // matrix[p, p] Wtcp = tcrossprod(W);

  // A = Wtcp + sigma2 * identity_matrix(p);
}
model {
  //prior on the first element of W being positive to make it identifiable
  // would be better to have a prior on the largest element. Maybe get that from the MLE and pass it in.

  // for (i in 1:k){
  //   target += std_normal_lpdf(W_raw[,k]);
  // } // I don't know why, but this does nothing to constrain W_raw.

  for (i in 1:p) {
    target += std_normal_lpdf(W_raw[i]);
  }

  target += exponential_lpdf(W_scales | 1);
  // W_scales ~ exponential(1);

  // This rarely works to set the sign of all the elements of W. Maybe pass in the sign of W_mle and add a huge negative penalty if it doesn't match?
  // for (i in 1:k) {
  //   target += inv_gamma_lpdf(W[mle_max_i[i],i] | 1,1);
  // }

  //
  //   for (i in 1:k) {
  //     for (j in 1:p) {
  //       // Stan is column major, double check that this is the right way to do this loop
  //       if (signnum(W[i,j] != W_signs[i,j])) {
  //         target += std_normal_lpdf(10);
  //       }
  //
  //     }
  //   }

  // target += std_normal_lpdf(W[,1]);
  // target += inv_gamma_lpdf(sqrt(W[1,1]^2 + W[2,1]^2)| 1,1);
  target += inv_gamma_lpdf(sigma2 | 1,1);

  target += w * (-(n - gamma * m) * 0.5 * log_mdl(sigma2, p, k, W) - 0.5 * trace(woodbury(W, sigma2, p, k) * C));
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
