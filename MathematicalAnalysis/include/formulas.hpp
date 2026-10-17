#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "matrix.hpp"

using ldouble = long double;
using ullong = unsigned long long;

// Basic characteristics of sample
ldouble average(const std::vector<ldouble>& values);
std::vector<ldouble> average_of_rows(const Matrix<ldouble>& matrix);
std::vector<ldouble> average_of_columns(const Matrix<ldouble>& matrix);
ldouble unbiased_point_estimate_of_variance(ldouble expected, const std::vector<ldouble>& values);
ldouble biased_central_statistical_moment_estimate(ldouble expected, ullong order, const std::vector<ldouble>& values);
Matrix<ldouble> biased_covariance_matrix(const Matrix<ldouble>& data);
Matrix<ldouble> unbiased_covariance_matrix(const Matrix<ldouble>& data);

// How fast CDF values-gaining-speed grows
ldouble gaussian_distribution_density(ldouble value, ldouble expected, ldouble sigma);

// Are values of the same Matrix collinear to each other
Matrix<ldouble> mahalanobis_distance(const Matrix<ldouble>& data);
// Squared Mahalanobis distances (Xi - X)^T S^-1 (Xj - X), S is the biased covariance matrix
// Mardia multivariate skewness (2.1) and kurtosis (2.2)
ldouble mardia_assymetry(const Matrix<ldouble>& data);
ldouble mardia_kurtosis(const Matrix<ldouble>& data);

// Get X2 by evaluating "rate" of random value insertion and compare whether it exceeds limit or not
ldouble sturges_formula(ldouble N);
ldouble brooks_caruthers_formula(ldouble N);
ldouble pearson_criteria(ldouble N, const std::vector<std::pair<ullong, ldouble>>& frequency_chance_pairs);
ldouble pearson_criteria_critical(ldouble significance_level, ldouble degrees_of_freedom);

// Upper quantile of the F distribution: x such that P(F > x) = significance_level
ldouble fisher_quantile(ldouble significance_level, ldouble d1, ldouble d2);

// Quantile of the standard normal law: x such that P(Z <= x) = probability
ldouble normal_quantile(ldouble probability);

// Compare real value distribution to expected
ldouble real_cumulative_distribution_function(const std::vector<ldouble>& data, ldouble value);
ldouble theoretical_cumulative_distribution_function(ldouble value, ldouble avg, ldouble rsmd);

// Get D and compare to C after transforming to D*
ldouble kolmogorov_smirnov_test(std::vector<ldouble> data);
bool stephens_normal_formula(ldouble d, std::size_t n, ldouble critical);
ldouble stephens_critical_value(ldouble alpha);

ldouble student_quantile(ldouble significance_level, ldouble degrees_of_freedom);

// If data is normal, find trusted ranges
std::pair<ldouble, ldouble> trusted_average_range(ldouble len, ldouble avg, ldouble deviation, ldouble t_quantile);
std::pair<ldouble, ldouble> trusted_deviation_range(ldouble len, ldouble deviation, ldouble significance);

// Half-width of the censoring interval by Grubbs (formula 1.2)
ldouble grubbs_delta(std::size_t N, ldouble deviation, ldouble significance);

// Half-width of the censoring interval for non-Gaussian data (formula 1.3)
ldouble non_gaussian_delta(std::size_t N, ldouble deviation, ldouble excess);

// Sample quantile by linear interpolation; values must be sorted ascending
ldouble sample_quantile(const std::vector<ldouble>& sorted_values, ldouble probability);

// Johnson normalizing transformation
enum class johnson_family { SL, SU, SB };

struct johnson_parameters {
    johnson_family family {johnson_family::SL};
    ldouble gamma {0};
    ldouble eta {0};
    ldouble epsilon {0};
    ldouble lambda {0};
    bool valid {false};
};

const char* johnson_family_name(johnson_family family);
johnson_parameters fit_johnson(std::vector<ldouble> data, ldouble z = 0.524L);
ldouble johnson_transform(ldouble x, const johnson_parameters& jp);
ldouble johnson_inverse_transform(ldouble z, const johnson_parameters& jp);

// Lab 3: squared Mahalanobis distance of every row to the mean, d_i^2 = (Zi - Z)^T S^-1 (Zi - Z),
// S is the biased covariance matrix (formula 3.9)
std::vector<ldouble> squared_mahalanobis_distances(const Matrix<ldouble>& data);

// Box-Cox transformation (3.6) of a single value
ldouble box_cox_transform(ldouble x, ldouble lambda);

// Multivariate Box-Cox: log-likelihood (3.8) for the vector of parameters lambda
// (one per column); -infinity when the data are not positive or S is degenerate
ldouble box_cox_log_likelihood(const Matrix<ldouble>& data, const std::vector<ldouble>& lambdas);

// Maximum likelihood estimate of the parameters lambda; empty when the data
// contain non-positive values and the transformation is not defined
std::vector<ldouble> fit_box_cox(const Matrix<ldouble>& data);

// Applies (3.6) column by column
Matrix<ldouble> apply_box_cox(const Matrix<ldouble>& data, const std::vector<ldouble>& lambdas);
