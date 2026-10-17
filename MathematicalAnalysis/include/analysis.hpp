#pragma once

#include "formulas.hpp"
#include <string>
#include <vector>

// Step 3: histogram of the sample with the fitted normal density drawn on it,
// written as a standalone SVG file
void write_histogram_svg(std::vector<ldouble> data, std::size_t sections_count,
                         const std::string& filename);

// Step 4: check whether the sample may be treated as Gaussian
bool gaussian_normality_check(std::vector<ldouble> data, ldouble significance);

// Step 6: censoring by Grubbs, assuming the data are Gaussian (formulas 1.1-1.2)
std::vector<ldouble> censor_normalized_data(std::vector<ldouble> data, ldouble significance);

// Step 6: censoring for non-Gaussian data via the Johnson normalizing
// transformation (formulas 1.4-1.6); falls back to formula (1.3) when the
// transformation cannot be fitted
std::vector<ldouble> censor_non_gaussian_data(std::vector<ldouble> data, ldouble significance);

// Dispatches to one of the two functions above
std::vector<ldouble> censor_data(std::vector<ldouble> data, bool isNormal, ldouble significance);

// Lab 2: check whether multivariate data (rows are observations, columns are
// variables) deviate from the normal law using Mardia skewness and kurtosis
bool mardia_normality_check(const Matrix<ldouble>& data, ldouble significance);


// Lab 3: Hotelling confidence ellipsoid of the sample means
// (X - m)^T S^-1 (X - m) <= k(N-1)/(N(N-k)) * F; prints the bound, the
// confidence interval of each mean (projection of the ellipsoid) and, for
// bivariate data, the semi-axes of the ellipse
void print_confidence_ellipsoid(const Matrix<ldouble>& data, ldouble significance);

// Right-hand side of the prediction ellipsoid equation (3.10): k(N^2-1)/(N(N-k)) * F
ldouble prediction_ellipsoid_border(std::size_t N, std::size_t k, ldouble significance);

// Bivariate data as a scatter plot with the confidence ellipse of the means and
// the prediction ellipse; the row `flagged` (if any) is highlighted as an outlier
void write_scatter_svg(const Matrix<ldouble>& data, std::size_t flagged, ldouble significance,
                       const std::string& title, const std::string& filename);
