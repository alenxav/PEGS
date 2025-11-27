
// C++ standard library headers
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// All R headers must be included within an extern "C" block
// when compiling with a C++ compiler.
extern "C" {
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
}

// Helper function for column-major indexing (as used by R and FORTRAN)
inline int idx(int row, int col, int num_rows) {
  return row + col * num_rows;
}

// Fisher-Yates shuffle using R's random number generator for reproducibility
void shuffle_indices(std::vector<int>& vec) {
  GetRNGstate();
  int n = static_cast<int>(vec.size());
  for (int i = n - 1; i > 0; i--) {
    int j = static_cast<int>(unif_rand() * (i + 1.0));
    std::swap(vec[i], vec[j]);
  }
  PutRNGstate();
}

extern "C" {
  
  SEXP PEGS_lapack(SEXP Y_R, SEXP X_R, SEXP maxit_R, SEXP logtol_R, SEXP NonNegativeCorr_R) {
    
    // --- 1. Get Input Dimensions and Parameters ---
    const int n0 = nrows(Y_R);
    const int k  = ncols(Y_R);
    const int p  = ncols(X_R);
    double *Y_ptr = REAL(Y_R);
    double *X_ptr = REAL(X_R);
    
    const int    maxit            = asInteger(maxit_R);
    const double logtol           = asReal(logtol_R);
    const bool   NonNegativeCorr  = asLogical(NonNegativeCorr_R);
    const int    one              = 1; // Used for BLAS calls
    
    // --- 2. Data Preparation and Memory Allocation ---
    std::vector<double> Y(n0 * k);
    std::vector<double> Z(n0 * k, 0.0);
    
    for (int i = 0; i < n0 * k; ++i) {
      if (ISNAN(Y_ptr[i])) {
        Y[i] = 0.0;
      } else {
        Y[i] = Y_ptr[i];
        Z[i] = 1.0;
      }
    }
    
    std::vector<double> n(k, 0.0);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n0; ++i) {
        n[j] += Z[idx(i, j, n0)];
      }
    }
    
    std::vector<double> iN_orig(k);
    for (int i = 0; i < k; ++i) iN_orig[i] = (n[i] > 0) ? 1.0 / n[i] : 0.0;
    
    std::vector<double> mu(k, 0.0);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n0; ++i) {
        mu[j] += Y[idx(i, j, n0)];
      }
      mu[j] *= iN_orig[j];
    }
    
    std::vector<double> y(n0 * k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n0; ++i) {
        y[idx(i, j, n0)] = (Y[idx(i, j, n0)] - mu[j]) * Z[idx(i, j, n0)];
      }
    }
    
    // --- 3. Pre-computation for Loop ---
    std::vector<double> XX(p * k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < p; ++i) {
        double sum_sq = 0.0;
        for (int r = 0; r < n0; ++r) {
          sum_sq += X_ptr[idx(r, i, n0)] * X_ptr[idx(r, i, n0)] * Z[idx(r, j, n0)];
        }
        XX[idx(i, j, p)] = sum_sq;
      }
    }
    
    std::vector<double> XSX(p * k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < p; ++i) {
        double sum_xz_in = 0.0;
        for (int r = 0; r < n0; ++r) {
          sum_xz_in += X_ptr[idx(r, i, n0)] * Z[idx(r, j, n0)];
        }
        sum_xz_in *= iN_orig[j];
        XSX[idx(i, j, p)] = XX[idx(i, j, p)] * iN_orig[j] - (sum_xz_in * sum_xz_in);
      }
    }
    
    std::vector<double> MSx(k, 0.0);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < p; ++i) MSx[j] += XSX[idx(i, j, p)];
    }
    
    std::vector<double> TrXSX(k);
    for (int i = 0; i < k; ++i) TrXSX[i] = n[i] * MSx[i];
    
    std::vector<double> iN(k);
    for (int i = 0; i < k; ++i) iN[i] = (n[i] > 1) ? 1.0 / (n[i] - 1.0) : 0.0;
    
    std::vector<double> vy(k);
    for (int j = 0; j < k; ++j) {
      double y_norm_sq = F77_CALL(ddot)(&n0, &y[j * n0], &one, &y[j * n0], &one);
      vy[j] = y_norm_sq * iN[j];
    }
    
    std::vector<double> ve = vy;
    for (double &val : ve) val *= 0.5;
    
    std::vector<double> iVe(k);
    for (int i = 0; i < k; ++i) iVe[i] = (ve[i] > 0) ? 1.0 / ve[i] : 0.0;
    
    std::vector<double> vb(k * k, 0.0);
    for (int i = 0; i < k; ++i) vb[idx(i, i, k)] = (MSx[i] > 0) ? ve[i] / MSx[i] : 0.0;
    
    std::vector<double> iG(k * k, 0.0);
    for (int i = 0; i < k; ++i) if (vb[idx(i, i, k)] > 0) iG[idx(i, i, k)] = 1.0 / vb[idx(i, i, k)];
    
    std::vector<double> tilde(p * k);
    char transa = 'T', transb = 'N';
    double alpha = 1.0, beta = 0.0;
    F77_CALL(dgemm)(&transa, &transb, &p, &k, &n0,
             &alpha,
             X_ptr, &n0,
             y.data(), &n0,
             &beta,
             tilde.data(), &p FCONE FCONE);
    
    // --- 4. Initialize Iteration Variables ---
    std::vector<double> b(p * k, 0.0);
    std::vector<double> e = y;
    std::vector<int> RGSvec(p);
    std::iota(RGSvec.begin(), RGSvec.end(), 0);
    
    double cnv = 10.0, inflate = 0.0;
    int numit = 0;
    
    // --- 5. Main Iteration Loop ---
    while (numit < maxit) {
      std::vector<double> beta0 = b;
      shuffle_indices(RGSvec);
      
      for (int j_idx = 0; j_idx < p; ++j_idx) {
        int J = RGSvec[j_idx];
        
        std::vector<double> b0(k);
        for (int i = 0; i < k; ++i) b0[i] = b[idx(J, i, p)];
        
        std::vector<double> LHS = iG;
        for (int i = 0; i < k; ++i) LHS[idx(i, i, k)] += XX[idx(J, i, p)] * iVe[i];
        
        std::vector<double> RHS(k);
        for (int i = 0; i < k; ++i) {
          RHS[i] = F77_CALL(ddot)(&n0, &X_ptr[J * n0], &one, &e[i * n0], &one);
          RHS[i] += XX[idx(J, i, p)] * b0[i];
          RHS[i] *= iVe[i];
        }
        
        char uplo = 'U'; 
        int info;
        F77_CALL(dpotrf)(&uplo, &k, LHS.data(), &k, &info FCONE);
        if (info != 0) {
          warning("Cholesky factorization failed in inner loop (marker %d). Skipping update.", J + 1);
          continue;
        }
        int nrhs = 1;
        F77_CALL(dpotrs)(&uplo, &k, &nrhs, LHS.data(), &k, RHS.data(), &k, &info FCONE);
        std::vector<double> b1 = RHS;
        
        for (int i = 0; i < k; ++i) {
          double delta_b = b1[i] - b0[i];
          b[idx(J, i, p)] = b1[i];
          if (fabs(delta_b) > 1e-12) { 
            double neg_delta_b = -delta_b;
            F77_CALL(daxpy)(&n0, &neg_delta_b, &X_ptr[J * n0], &one, &e[i * n0], &one);
          }
        }
      }
      
      for (int j = 0; j < k; ++j) {
        double ve_sum = F77_CALL(ddot)(&n0, &e[j * n0], &one, &y[j * n0], &one);
        ve[j]  = ve_sum * iN[j];
        iVe[j] = (ve[j] > 0) ? 1.0 / ve[j] : 0.0;
      }
      
      std::vector<double> TildeHat(k * k);
      F77_CALL(dgemm)(&transa, &transb, &k, &k, &p,
               &alpha,
               b.data(), &p,
               tilde.data(), &p,
               &beta,
               TildeHat.data(), &k FCONE FCONE);
      
      for (int c = 0; c < k; ++c) {
        for (int r = c; r < k; ++r) {
          if (r == c) {
            vb[idx(r, c, k)] = (TrXSX[r] > 0) ? TildeHat[idx(r, c, k)] / TrXSX[r] : 0.0;
          } else {
            double denom = TrXSX[r] + TrXSX[c];
            double val = (denom > 0) ? (TildeHat[idx(r, c, k)] + TildeHat[idx(c, r, k)]) / denom : 0.0;
            vb[idx(r, c, k)] = vb[idx(c, r, k)] = val;
          }
        }
      }
      
      if (NonNegativeCorr) {
        for (int i = 0; i < k * k; ++i) if (vb[i] < 0.0) vb[i] = 0.0;
      }
      
      std::vector<double> vb_copy = vb;
      std::vector<double> eigvals(k);
      char jobz = 'N', uplo_eig = 'U'; 
      int info_eig;
      int lwork = std::max(1, 3 * k - 1); 
      std::vector<double> work(lwork);
      F77_CALL(dsyev)(&jobz, &uplo_eig, &k, 
               vb_copy.data(), &k, 
               eigvals.data(), 
               work.data(), &lwork, 
               &info_eig FCONE FCONE);
      
      double MinDVb = eigvals[0];
      for (int i = 1; i < k; ++i) if (eigvals[i] < MinDVb) MinDVb = eigvals[i];
      
      if (MinDVb < 0.001) {
        double new_inflate = fabs(MinDVb * 1.1);
        if (new_inflate > inflate) inflate = new_inflate;
      }
      for (int i = 0; i < k; ++i) vb[idx(i, i, k)] += inflate;
      
      vb_copy = vb;
      char uplo_inv = 'U'; 
      int info_inv;
      F77_CALL(dpotrf)(&uplo_inv, &k, vb_copy.data(), &k, &info_inv FCONE);
      if (info_inv == 0) {
        F77_CALL(dpotri)(&uplo_inv, &k, vb_copy.data(), &k, &info_inv FCONE);
        if (info_inv == 0) {
          for (int c = 0; c < k; ++c)
            for (int r = c + 1; r < k; ++r)
              vb_copy[idx(r, c, k)] = vb_copy[idx(c, r, k)];
          iG = vb_copy;
        }
      }
      
      std::vector<double> b0_e(k, 0.0);
      for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n0; ++i) b0_e[j] += e[idx(i, j, n0)];
        b0_e[j] *= iN_orig[j]; // Use original iN for this part
        mu[j] += b0_e[j];
      }
      for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n0; ++i) {
          e[idx(i, j, n0)] = (e[idx(i, j, n0)] - b0_e[j]) * Z[idx(i, j, n0)];
        }
      }
      
      double diff_sum_sq = 0.0;
      for (size_t i = 0; i < b.size(); ++i) {
        double diff = beta0[i] - b[i];
        diff_sum_sq += diff * diff;
      }
      cnv = log10(diff_sum_sq);
      ++numit;
      if (numit % 100 == 0) { Rprintf("Iter: %d || Conv: %f\n", numit, cnv); }
      if (cnv < logtol) break;
    }
    
    // --- 6. Final Calculations and Output ---
    std::vector<double> h2(k);
    for (int i = 0; i < k; ++i) h2[i] = (vy[i] > 0) ? 1.0 - ve[i] / vy[i] : 0.0;
    
    std::vector<double> hat(n0 * k, 0.0);
    char transb_final = 'N';
    F77_CALL(dgemm)(&transb_final, &transb_final, &n0, &k, &p,
             &alpha,
             X_ptr, &n0,
             b.data(), &p,
             &beta,
             hat.data(), &n0 FCONE FCONE);
    for (int j = 0; j < k; ++j) 
      for (int i = 0; i < n0; ++i) 
        hat[idx(i, j, n0)] += mu[j];
    
    std::vector<double> GC(k * k, 0.0);
    for (int c = 0; c < k; ++c) {
      for (int r = c; r < k; ++r) {
        double sd_r = sqrt(vb[idx(r, r, k)]);
        double sd_c = sqrt(vb[idx(c, c, k)]);
        if (sd_r > 0 && sd_c > 0) {
          double val = vb[idx(r, c, k)] / (sd_r * sd_c);
          GC[idx(r, c, k)] = val;
          GC[idx(c, r, k)] = val;
        } else {
          GC[idx(r, c, k)] = 0.0;
          GC[idx(c, r, k)] = 0.0;
        }
      }
    }
    
    // --- 7. Create R List for Output ---
    const char *names[] = {"mu", "b", "hat", "h2", "GC", "bend", "numit", "cnv", ""};
    SEXP res = PROTECT(mkNamed(VECSXP, names));
    
    SEXP mu_R = PROTECT(allocVector(REALSXP, k));
    if (k > 0) std::copy(mu.begin(), mu.end(), REAL(mu_R));
    SET_VECTOR_ELT(res, 0, mu_R);
    
    SEXP b_R = PROTECT(allocMatrix(REALSXP, p, k));
    if (p * k > 0) std::copy(b.begin(), b.end(), REAL(b_R));
    SET_VECTOR_ELT(res, 1, b_R);
    
    SEXP hat_R = PROTECT(allocMatrix(REALSXP, n0, k));
    if (n0 * k > 0) std::copy(hat.begin(), hat.end(), REAL(hat_R));
    SET_VECTOR_ELT(res, 2, hat_R);
    
    SEXP h2_R = PROTECT(allocVector(REALSXP, k));
    if (k > 0) std::copy(h2.begin(), h2.end(), REAL(h2_R));
    SET_VECTOR_ELT(res, 3, h2_R);
    
    SEXP GC_R = PROTECT(allocMatrix(REALSXP, k, k));
    if (k * k > 0) std::copy(GC.begin(), GC.end(), REAL(GC_R));
    SET_VECTOR_ELT(res, 4, GC_R);
    
    SET_VECTOR_ELT(res, 5, ScalarReal(inflate));
    SET_VECTOR_ELT(res, 6, ScalarInteger(numit));
    SET_VECTOR_ELT(res, 7, ScalarReal(cnv));
    
    UNPROTECT(6); // res, mu_R, b_R, hat_R, h2_R, GC_R
    return res;
  }
  
} // extern "C"
