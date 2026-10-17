#include "formulas.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

ldouble average(const std::vector<ldouble>& values) {
    ldouble sum {0};
    for (auto v : values) {
        sum += v;
    }
    return sum / values.size();
}

std::vector<ldouble> average_of_rows(const Matrix<ldouble>& matrix) {
    std::vector<ldouble> result;
    for (size_t i=0; i<matrix.rows_count(); ++i) {
        ldouble avg {0};
        for (size_t j=0; j<matrix.columns_count(); ++j) {
            avg += matrix[i,j]/matrix.columns_count();
        }
        result.push_back(avg);
    }
    return result;
}

std::vector<ldouble> average_of_columns(const Matrix<ldouble>& matrix) {
    std::vector<ldouble> result;
    for (size_t i=0; i<matrix.columns_count(); ++i) {
        ldouble avg {0};
        for (size_t j=0; j<matrix.rows_count(); ++j) {
            avg += matrix[j,i]/matrix.rows_count();
        }
        result.push_back(avg);
    }
    return result;
}

ldouble unbiased_point_estimate_of_variance(ldouble expected, const std::vector<ldouble>& values) {
    ldouble result {0};
    for (auto v : values) {
        result += std::pow(v - expected, 2)/(values.size()-1);
    }
    return result;
}

ldouble biased_central_statistical_moment_estimate(ldouble expected, ullong order, const std::vector<ldouble>& values) {
    ldouble result {0};
    for (auto v : values) {
        result += std::pow(v - expected, order)/values.size();
    }
    return result;
}

Matrix<ldouble> biased_covariance_matrix(const Matrix<ldouble>& data) {
    Matrix<ldouble> result {data.columns_count(), data.columns_count()};
    Matrix<ldouble> avgs {{average_of_columns(data)}};
    for (size_t i=0; i<data.rows_count(); ++i) {
        auto raw = Matrix<ldouble>{{data.get_row(i)}} - avgs;
        auto step = (*raw) * raw;
        result += step/data.rows_count();
    }
    return result;
}

Matrix<ldouble> unbiased_covariance_matrix(const Matrix<ldouble>& data) {
    Matrix<ldouble> result {data.columns_count(), data.columns_count()};
    Matrix<ldouble> avgs {{average_of_columns(data)}};
    for (size_t i=0; i<data.rows_count(); ++i) {
        auto raw = Matrix<ldouble>{{data.get_row(i)}} - avgs;
        auto step = (*raw) * raw;
        result += step/(data.rows_count()-1);
    }
    return result;
}

ldouble gaussian_distribution_density(ldouble value, ldouble expected, ldouble sigma) {
    ldouble pow = std::pow(value - expected, 2)/(2*sigma*sigma);
    ldouble exp = std::exp(-pow);
    ldouble denom = sigma * std::sqrt(2*M_PI);
    return exp/denom;
}

Matrix<ldouble> mahalanobis_distance(const Matrix<ldouble> &data) {
    const size_t n = data.rows_count();
    Matrix<ldouble> result {n, n};
    Matrix<ldouble> average {{average_of_columns(data)}};
    Matrix<ldouble> inv_cov = biased_covariance_matrix(data).pow(-1);

    // Отклонения считаем один раз, а не N² раз
    std::vector<Matrix<ldouble>> d;
    d.reserve(n);
    for (size_t i = 0; i < n; ++i)
        d.push_back(Matrix<ldouble>{{data.get_row(i)}} - average);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            result[i,j] = (d[i] * inv_cov * (*d[j])).get(0, 0);
    return result;
}
ldouble mardia_assymetry(const Matrix<ldouble>& data) {
    const size_t n = data.rows_count();
    // Squared Mahalanobis distances (Xi - X)^T S^-1 (Xj - X) between all pairs
    Matrix<ldouble> distances = mahalanobis_distance(data);
    ldouble sum {0};
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            sum += std::pow(distances[i,j], 3);
    return sum/(ldouble(n)*n);
}

ldouble mardia_kurtosis(const Matrix<ldouble>& data) {
    const size_t n = data.rows_count();
    Matrix<ldouble> distances = mahalanobis_distance(data);
    ldouble sum {0};
    for (size_t i = 0; i < n; ++i)
        sum += std::pow(distances[i,i], 2);
    return sum/n;
}

ldouble sturges_formula(ldouble N) {
    return std::log2(N)+1;
}

ldouble brooks_caruthers_formula(ldouble N) {
    return 5*std::log10(N);
}

ldouble pearson_criteria(ldouble N, const std::vector<std::pair<ullong, ldouble>>& frequency_chance_pairs) {
    ldouble result {};
    for (auto p : frequency_chance_pairs) {
        ldouble Np = N*p.second;
        if (Np <= 0) continue;
        result += std::pow(ldouble(p.first) - Np, 2)/Np;
    }
    return result;
}

namespace {

ldouble gamma_series(ldouble a, ldouble x) {
    constexpr int max_iterations {200};
    constexpr ldouble epsilon {1e-14L};

    ldouble term {1/a};
    ldouble sum {term};
    ldouble ap {a};
    for (int n = 0; n < max_iterations; ++n) {
        ap += 1;
        term *= x/ap;
        sum += term;
        if (std::abs(term) < std::abs(sum)*epsilon) break;
    }
    return sum*std::exp(-x + a*std::log(x) - std::lgamma(a));
}

ldouble gamma_continued_fraction(ldouble a, ldouble x) {
    constexpr int max_iterations {200};
    constexpr ldouble epsilon {1e-14L};
    constexpr ldouble tiny {1e-300L};

    ldouble b {x + 1 - a};
    ldouble c {1/tiny};
    ldouble d {1/b};
    ldouble h {d};
    for (int i = 1; i <= max_iterations; ++i) {
        ldouble an {-static_cast<ldouble>(i)*(i - a)};
        b += 2;
        d = an*d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an/c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1/d;
        ldouble delta {d*c};
        h *= delta;
        if (std::abs(delta - 1) < epsilon) break;
    }
    return std::exp(-x + a*std::log(x) - std::lgamma(a))*h;
}

// Regularized lower incomplete gamma function P(a, x); the series form
// converges quickly for x < a+1, the continued fraction for x >= a+1.
ldouble regularized_lower_incomplete_gamma(ldouble a, ldouble x) {
    if (x <= 0) return 0;
    if (x < a + 1)
        return gamma_series(a, x);
    else
        return 1 - gamma_continued_fraction(a, x);
}

ldouble beta_continued_fraction(ldouble a, ldouble b, ldouble x) {
    constexpr int max_iterations {300};
    constexpr ldouble epsilon {1e-14L};
    constexpr ldouble tiny {1e-300L};

    ldouble qab {a + b}, qap {a + 1}, qam {a - 1};
    ldouble c {1};
    ldouble d {1 - qab*x/qap};
    if (std::abs(d) < tiny) d = tiny;
    d = 1/d;
    ldouble h {d};

    for (int m = 1; m <= max_iterations; ++m) {
        ldouble m2 {static_cast<ldouble>(2*m)};

        ldouble aa {m*(b - m)*x/((qam + m2)*(a + m2))};
        d = 1 + aa*d; if (std::abs(d) < tiny) d = tiny;
        c = 1 + aa/c; if (std::abs(c) < tiny) c = tiny;
        d = 1/d;
        h *= d*c;

        aa = -(a + m)*(qab + m)*x/((a + m2)*(qap + m2));
        d = 1 + aa*d; if (std::abs(d) < tiny) d = tiny;
        c = 1 + aa/c; if (std::abs(c) < tiny) c = tiny;
        d = 1/d;
        ldouble delta {d*c};
        h *= delta;

        if (std::abs(delta - 1) < epsilon) break;
    }
    return h;
}

// Regularized incomplete beta function I_x(a, b).
ldouble regularized_incomplete_beta(ldouble x, ldouble a, ldouble b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;

    ldouble front {std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
        + a*std::log(x) + b*std::log(1 - x))};

    if (x < (a + 1)/(a + b + 2))
        return front*beta_continued_fraction(a, b, x)/a;
    else
        return 1 - front*beta_continued_fraction(b, a, 1 - x)/b;
}

ldouble chi_squared_cdf(ldouble x, ldouble degrees_of_freedom) {
    if (x <= 0) return 0;
    return regularized_lower_incomplete_gamma(degrees_of_freedom/2, x/2);
}

ldouble student_t_cdf(ldouble t, ldouble degrees_of_freedom) {
    ldouble x {degrees_of_freedom/(degrees_of_freedom + t*t)};
    ldouble ib {regularized_incomplete_beta(x, degrees_of_freedom/2, 0.5L)};
    return t > 0 ? 1 - ib/2 : ib/2;
}

// Bisection root-find for the x where a monotonically increasing cdf crosses
// target, expanding [lo, hi] outward first until it brackets the root.
template<typename Cdf>
ldouble invert_monotonic_cdf(Cdf cdf, ldouble target, ldouble lo, ldouble hi) {
    constexpr int max_iterations {200};
    constexpr int max_expansions {200};

    for (int i = 0; i < max_expansions && cdf(lo) > target; ++i) lo *= 2;
    for (int i = 0; i < max_expansions && cdf(hi) < target; ++i) hi *= 2;

    for (int i = 0; i < max_iterations; ++i) {
        ldouble mid {(lo + hi)/2};
        if (cdf(mid) < target) lo = mid; else hi = mid;
    }
    return (lo + hi)/2;
}

ldouble quiet_nan() {
    return std::numeric_limits<ldouble>::quiet_NaN();
}

} // namespace

ldouble normal_quantile(ldouble probability) {
    auto cdf = [](ldouble x) { return std::erfc(-x/std::sqrt(2.0L))/2; };
    return invert_monotonic_cdf(cdf, probability, -1.0L, 1.0L);
}

ldouble pearson_criteria_critical(ldouble significance_level, ldouble degrees_of_freedom) {
    auto cdf = [degrees_of_freedom](ldouble x) { return chi_squared_cdf(x, degrees_of_freedom); };
    return invert_monotonic_cdf(cdf, 1 - significance_level, 0.0L, 1.0L);
}

ldouble fisher_quantile(ldouble significance_level, ldouble d1, ldouble d2) {
    auto cdf = [d1, d2](ldouble x) {
        if (x <= 0) return ldouble{0};
        return regularized_incomplete_beta(d1*x/(d1*x + d2), d1/2, d2/2);
    };
    return invert_monotonic_cdf(cdf, 1 - significance_level, 0.0L, 1.0L);
}

ldouble real_cumulative_distribution_function(const std::vector<ldouble>& data, ldouble value) {
    ldouble result = 0;
    for (auto v : data) {
        if (v > value) continue;
        result++;
    }
    return result/ldouble(data.size());
}

ldouble theoretical_cumulative_distribution_function(ldouble value, ldouble avg, ldouble rsmd) {
    if (rsmd == 0) return 0;
    ldouble v = (value - avg) / rsmd;
    ldouble result = 0.5L * std::erfc(-v / std::sqrt(2.0L));
    return result;
}

ldouble kolmogorov_smirnov_test(std::vector<ldouble> data) {
    std::sort(data.begin(), data.end());

    ldouble n = data.size();
    ldouble avg = average(data);
    ldouble rsmd = std::sqrt(unbiased_point_estimate_of_variance(avg, data));
    if (rsmd == 0) return 1;

    ldouble dP = 0, dM = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        ldouble k = static_cast<ldouble>(i);
        ldouble cdf = theoretical_cumulative_distribution_function(data[i], avg, rsmd);
        dP = std::max(dP, (k + 1)/n - cdf);
        dM = std::max(dM, cdf - k/n);
    }
    return std::max(dP, dM);
}

bool stephens_normal_formula(ldouble d, std::size_t n, ldouble critical) {
    ldouble sn = std::sqrt(static_cast<ldouble>(n));
    return d * (sn - 0.01L + 0.85L / sn) <= critical;
}

ldouble stephens_critical_value(ldouble alpha) {
    static constexpr ldouble a[] = {0.15L,  0.10L,  0.05L,  0.025L, 0.01L};
    static constexpr ldouble d[] = {0.775L, 0.819L, 0.895L, 0.955L, 1.035L};
    constexpr std::size_t m = 5;

    if (alpha >= a[0])     return d[0];
    if (alpha <= a[m - 1]) return d[m - 1];

    for (std::size_t k = 0; k + 1 < m; ++k) {
        if (alpha > a[k + 1]) {
            ldouble t = (std::log(alpha)      - std::log(a[k])) /
                        (std::log(a[k + 1])   - std::log(a[k]));
            return d[k] + t * (d[k + 1] - d[k]);
        }
    }
    return d[m - 1];
}

ldouble student_quantile(ldouble significance_level, ldouble degrees_of_freedom) {
    auto cdf = [degrees_of_freedom](ldouble t) { return student_t_cdf(t, degrees_of_freedom); };
    return invert_monotonic_cdf(cdf, 1 - significance_level, -1.0L, 1.0L);
}

std::pair<ldouble, ldouble> trusted_average_range(ldouble len, ldouble avg, ldouble deviation, ldouble t_quantile) {
    return {
        avg - t_quantile*deviation/std::sqrt(len),
        avg + t_quantile*deviation/std::sqrt(len),
    };
}

std::pair<ldouble, ldouble> trusted_deviation_range(ldouble len, ldouble deviation, ldouble significance) {
    return  {
        deviation*std::sqrt(len/pearson_criteria_critical(significance/2, len-1)),
        deviation*std::sqrt(len/pearson_criteria_critical(1-(significance/2), len-1)),
    };
}

// Formula (1.2): delta = S * (N-1)/sqrt(N) * sqrt(t^2 / (N - 2 + t^2)),
// where t is the Student quantile for alpha/(2N) and N-2 degrees of freedom.
ldouble grubbs_delta(std::size_t N, ldouble deviation, ldouble significance) {
    if (N < 3) return 0;
    ldouble n = static_cast<ldouble>(N);
    ldouble t = student_quantile(significance/(2*n), n - 2);
    ldouble t2 = t*t;
    return deviation * ((n - 1)/std::sqrt(n)) * std::sqrt(t2/(n - 2 + t2));
}

// Formula (1.3): delta_x = [1.55 + 0.8*sqrt(excess - 1)*lg(N/10)] * S_x
ldouble non_gaussian_delta(std::size_t N, ldouble deviation, ldouble excess) {
    ldouble n = static_cast<ldouble>(N);
    ldouble root = excess > 1 ? std::sqrt(excess - 1) : 0;
    return deviation * (1.55L + 0.8L*root*std::log10(n/10));
}

ldouble sample_quantile(const std::vector<ldouble>& sorted_values, ldouble probability) {
    std::size_t n = sorted_values.size();
    if (n == 0) return 0;
    if (n == 1) return sorted_values.front();

    ldouble position = probability * (n + 1);   // ranks are counted from one
    if (position <= 1) return sorted_values.front();
    if (position >= n) return sorted_values.back();

    std::size_t k = static_cast<std::size_t>(position);
    ldouble fraction = position - static_cast<ldouble>(k);
    return sorted_values[k-1] + fraction*(sorted_values[k] - sorted_values[k-1]);
}

const char* johnson_family_name(johnson_family family) {
    switch (family) {
        case johnson_family::SL: return "SL (lognormal)";
        case johnson_family::SU: return "SU (unbounded)";
        default:                 return "SB (bounded)";
    }
}

// Slifker & Shapiro percentile method: four sample quantiles taken at
// -3z, -z, z and 3z on the normal scale decide the family and its parameters.
johnson_parameters fit_johnson(std::vector<ldouble> data, ldouble z) {
    johnson_parameters jp {};
    if (data.size() < 4) return jp;

    std::sort(data.begin(), data.end());

    ldouble phi1 = theoretical_cumulative_distribution_function(z,   0.0L, 1.0L);
    ldouble phi3 = theoretical_cumulative_distribution_function(3*z, 0.0L, 1.0L);

    ldouble x_minus3 = sample_quantile(data, 1 - phi3);
    ldouble x_minus1 = sample_quantile(data, 1 - phi1);
    ldouble x_plus1  = sample_quantile(data, phi1);
    ldouble x_plus3  = sample_quantile(data, phi3);

    ldouble m = x_plus3  - x_plus1;   // length of the right tail
    ldouble n = x_minus1 - x_minus3;  // length of the left tail
    ldouble p = x_plus1  - x_minus1;  // width of the central part

    if (m <= 0 || n <= 0 || p <= 0) return jp;

    ldouble center = 0.5L*(x_plus1 + x_minus1);
    ldouble discriminant = m*n/(p*p);

    constexpr ldouble tolerance = 0.05L;

    if (std::abs(discriminant - 1) < tolerance) {
        ldouble mp = m/p;
        if (mp <= 1) return jp;
        jp.family  = johnson_family::SL;
        jp.eta     = 2*z/std::log(mp);
        jp.gamma   = jp.eta*std::log((mp - 1)/(p*std::sqrt(mp)));
        jp.epsilon = center - 0.5L*p*(mp + 1)/(mp - 1);
        jp.lambda  = 1;                       // unused by this family
    } else if (discriminant > 1) {
        ldouble mp = m/p, np = n/p;
        ldouble sum = mp + np;
        if (sum <= 2 || mp*np <= 1) return jp;
        ldouble root = std::sqrt(mp*np - 1);
        jp.family  = johnson_family::SU;
        jp.eta     = 2*z/std::acosh(0.5L*sum);
        jp.gamma   = jp.eta*std::asinh((np - mp)/(2*root));
        jp.lambda  = 2*p*root/((sum - 2)*std::sqrt(sum + 2));
        jp.epsilon = center + p*(np - mp)/(2*(sum - 2));
    } else {
        ldouble pm = p/m, pn = p/n;
        ldouble product = (1 + pm)*(1 + pn);
        ldouble denominator = p*p/(m*n) - 1;
        if (product <= 4 || denominator == 0) return jp;
        ldouble inner = (product - 2)*(product - 2) - 4;
        if (inner < 0) return jp;
        jp.family  = johnson_family::SB;
        jp.eta     = z/std::acosh(0.5L*std::sqrt(product));
        jp.gamma   = jp.eta*std::asinh((pn - pm)*std::sqrt(product - 4)/(2*denominator));
        jp.lambda  = p*std::sqrt(inner)/denominator;
        jp.epsilon = center - jp.lambda/2 + p*(pn - pm)/(2*denominator);
    }

    jp.valid = std::isfinite(jp.eta) && std::isfinite(jp.gamma)
            && std::isfinite(jp.epsilon) && std::isfinite(jp.lambda)
            && jp.eta != 0;
    return jp;
}

// Formula (1.4): z = psi(x)
ldouble johnson_transform(ldouble x, const johnson_parameters& jp) {
    if (!jp.valid) return quiet_nan();

    ldouble shifted = x - jp.epsilon;
    switch (jp.family) {
        case johnson_family::SL:
            if (shifted <= 0) return quiet_nan();
            return jp.gamma + jp.eta*std::log(shifted);
        case johnson_family::SU:
            if (jp.lambda <= 0) return quiet_nan();
            return jp.gamma + jp.eta*std::asinh(shifted/jp.lambda);
        default: // SB
            if (shifted <= 0 || shifted >= jp.lambda) return quiet_nan();
            return jp.gamma + jp.eta*std::log(shifted/(jp.lambda - shifted));
    }
}

// Formula (1.6): x = psi^(-1)(z)
ldouble johnson_inverse_transform(ldouble z, const johnson_parameters& jp) {
    if (!jp.valid) return quiet_nan();

    ldouble u = (z - jp.gamma)/jp.eta;
    switch (jp.family) {
        case johnson_family::SL:
            return jp.epsilon + std::exp(u);
        case johnson_family::SU:
            return jp.epsilon + jp.lambda*std::sinh(u);
        default: // SB
            return jp.epsilon + jp.lambda/(1 + std::exp(-u));
    }
}


std::vector<ldouble> squared_mahalanobis_distances(const Matrix<ldouble>& data) {
    const size_t n = data.rows_count();
    Matrix<ldouble> average {{average_of_columns(data)}};
    Matrix<ldouble> inv_cov = biased_covariance_matrix(data).inverse();

    std::vector<ldouble> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Matrix<ldouble> d = Matrix<ldouble>{{data.get_row(i)}} - average;
        result.push_back((d * inv_cov * (*d)).get(0, 0));
    }
    return result;
}

ldouble box_cox_transform(ldouble x, ldouble lambda) {
    if (lambda == 0) return std::log(x);
    return (std::pow(x, lambda) - 1)/lambda;
}

Matrix<ldouble> apply_box_cox(const Matrix<ldouble>& data, const std::vector<ldouble>& lambdas) {
    Matrix<ldouble> result {data.rows_count(), data.columns_count()};
    for (size_t i = 0; i < data.rows_count(); ++i)
        for (size_t j = 0; j < data.columns_count(); ++j)
            result[i, j] = box_cox_transform(data[i, j], lambdas[j]);
    return result;
}

ldouble box_cox_log_likelihood(const Matrix<ldouble>& data, const std::vector<ldouble>& lambdas) {
    constexpr ldouble impossible {-std::numeric_limits<ldouble>::infinity()};
    const size_t n = data.rows_count();

    ldouble sum {0};
    for (size_t j = 0; j < data.columns_count(); ++j) {
        ldouble log_sum {0};
        for (size_t i = 0; i < n; ++i) {
            if (data[i, j] <= 0) return impossible;
            log_sum += std::log(data[i, j]);
        }
        sum += (lambdas[j] - 1)*log_sum;
    }

    ldouble det = biased_covariance_matrix(apply_box_cox(data, lambdas)).determinant();
    if (!(det > 0) || !std::isfinite(det)) return impossible;
    return sum - static_cast<ldouble>(n)/2*std::log(det);
}

std::vector<ldouble> fit_box_cox(const Matrix<ldouble>& data) {
    const size_t k = data.columns_count();
    for (size_t i = 0; i < data.rows_count(); ++i)
        for (size_t j = 0; j < k; ++j)
            if (data[i, j] <= 0) return {};

    constexpr ldouble lower {-5}, upper {5};
    const ldouble phi {(std::sqrt(5.0L) - 1)/2};
    std::vector<ldouble> lambdas(k, 1.0L);

    // Coordinate-wise golden section search; the likelihood is smooth in each
    // parameter, so sweeping over the parameters until they settle is enough
    for (int sweep = 0; sweep < 50; ++sweep) {
        ldouble moved {0};
        for (size_t j = 0; j < k; ++j) {
            auto value = [&](ldouble l) {
                std::vector<ldouble> trial = lambdas;
                trial[j] = l;
                return box_cox_log_likelihood(data, trial);
            };
            ldouble a {lower}, b {upper};
            ldouble c {b - phi*(b - a)}, d {a + phi*(b - a)};
            ldouble fc {value(c)}, fd {value(d)};
            for (int it = 0; it < 80; ++it) {
                if (fc > fd) {
                    b = d; d = c; fd = fc;
                    c = b - phi*(b - a); fc = value(c);
                } else {
                    a = c; c = d; fc = fd;
                    d = a + phi*(b - a); fd = value(d);
                }
            }
            ldouble best = (a + b)/2;
            moved = std::max(moved, std::abs(best - lambdas[j]));
            lambdas[j] = best;
        }
        if (moved < 1e-9L) break;
    }
    return lambdas;
}
