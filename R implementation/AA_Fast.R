"
Implementation of Frank-Wolfe algorithm for the archetypal analysis.

Algorithm is based on the paper: 'Archetypal Analysis as an Autoencoder'
(https://www.researchgate.net/publication/282733207_Archetypal_Analysis_as_an_Autoencoder)

Author: Guillermo García Cobo
" 

library('Matrix')
require('ramify')
library('ramify')

computeA <- function(X, Z, n_archetypes, n_samples, derivative_max_iter) {
  A = matrix(0., n_archetypes, n_samples)
  A[0,] = 1.
  e = matrix(0., n_archetypes, n_samples)
  
  for (t in 0:derivative_max_iter-1) {
    G = 2. * (crossprod(Z, Z) %*% A - crossprod(Z, X))
    argmins = ramify::argmin(G, rows=F)
    e[cbind(argmins, 1:n_samples)] = 1.0
    A = A + 2.0/(t + 2.0) * (e - A) 
    e[cbind(argmins, 1:n_samples)] = 0.0
  }
  
  return(A)
}

computeB <- function(X, A, n_archetypes, n_samples, derivative_max_iter) {
  B = matrix(0., n_samples, n_archetypes)
  B[0,] = 1.
  e = matrix(0., n_samples, n_archetypes)
  
  for (t in 0:derivative_max_iter-1) {
    G = 2. * (crossprod(X, (X %*% B) %*% tcrossprod(A, A)) - crossprod(X, tcrossprod(X, A)))
    argmins = ramify::argmin(G, rows=F)
    e[cbind(argmins, 1:n_archetypes)] = 1.0
    B = B + 2.0/(t + 2.0) * (e - B) 
    e[cbind(argmins, 1:n_archetypes)] = 0.0
  }
  
  return(B)
}

aa_fast <- function(X, n_archetypes, max_iter=100, tol=1E-10, verbose=F, derivative_max_iter=10) {
  n_samples = NROW(X)
  n_features = NCOL(X)
  
  X = t(X)
  
  B = diag(1., n_samples, n_archetypes)
  Z = X %*% B
  
  prev_RSS = NA
  
  for (iter in 1:max_iter) {
    A = computeA(X, Z, n_archetypes, n_samples, derivative_max_iter)
    B = computeB(X, A, n_archetypes, n_samples, derivative_max_iter)
    
    Z = X %*% B
    
    RSS = sum((X - Z %*% A)^2)
    if (!is.na(prev_RSS) && abs(prev_RSS - RSS) / prev_RSS < tol) {
      break
    }
    prev_RSS = RSS
    
  }
  
  return(c(t(Z), RSS))
}
