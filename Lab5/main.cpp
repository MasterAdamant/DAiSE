//Containers
#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

//File reading and logging
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

//Math
#include "formulas.hpp"
#include "analysis.hpp"
#include "matrix.hpp"
#include "data_reader.hpp"

namespace {

// Mutually inverse normalizing transformation (5.1)-(5.2) of the pair (X1, Y)
struct Normalization {
    std::string name;
    std::string tag;                                    // suffix of the output files
    std::function<ldouble(ldouble)> forward_x;          // psi_1
    std::function<ldouble(ldouble)> forward_y;          // psi_Y
    std::function<ldouble(ldouble)> inverse_y;          // psi_Y^-1
    std::function<std::string(ldouble, ldouble)> equation;  // nonlinear equation (5.4) for b0, b1
};

struct Quality {
    ldouble r2 {0}, mmre {0}, pred {0};
};

struct NonlinearResult {
    std::string name;
    std::string equation;
    Quality quality;
    bool residuals_normal {false};
};

std::string Number(ldouble value) {
    std::ostringstream out;
    out << double(value);
    return out.str();
}

Quality Evaluate(const std::vector<ldouble>& y, const std::vector<ldouble>& predicted) {
    std::vector<ldouble> mre = magnitudes_of_relative_error(y, predicted);
    return {determination_coefficient(y, predicted), mean_magnitude_of_relative_error(mre),
            percentage_of_prediction(mre, 0.25L)};
}

// Characteristics of the deviations: the mean must be zero for the least squares
// estimate with an intercept, sigma^2 = SSE/(N-2) estimates the variance of the error
void PrintDeviationCharacteristics(const linear_regression& model) {
    const std::vector<ldouble>& e = model.residuals;
    ldouble avg = average(e);
    ldouble sigma = std::sqrt(model.sse/static_cast<ldouble>(e.size() - 2));
    ldouble s = std::sqrt(biased_central_statistical_moment_estimate(avg, 2, e));
    ldouble skewness = biased_central_statistical_moment_estimate(avg, 3, e)/std::pow(s, 3);
    ldouble excess = biased_central_statistical_moment_estimate(avg, 4, e)/std::pow(s, 4) - 3;
    auto [lo, hi] = std::minmax_element(e.begin(), e.end());
    std::size_t positive = 0, far = 0;
    for (auto v : e) {
        if (v > 0) ++positive;
        if (std::abs(v) > 2*sigma) ++far;
    }
    std::cout << "Sum of deviations: " << double(avg*e.size()) << "\n"
              << "SSE (sum of squared deviations): " << double(model.sse) << "\n"
              << "Standard deviation sigma = sqrt(SSE/(N-2)): " << double(sigma) << "\n"
              << "Min / max deviation: " << double(*lo) << " / " << double(*hi) << "\n"
              << "Positive / negative deviations: " << positive << " / " << e.size() - positive << "\n"
              << "Skewness: " << double(skewness) << ", excess: " << double(excess) << " (0 and 0 for the normal law)\n"
              << "Deviations with |e| > 2*sigma: " << far << "\n";
}

void PrintQuality(const Quality& q) {
    std::cout << "R^2 = " << double(q.r2) << (q.r2 > 0.75L ? " (acceptable, > 0.75)" : " (NOT acceptable, <= 0.75)") << "\n"
              << "MMRE = " << double(q.mmre) << (q.mmre <= 0.25L ? " (acceptable, <= 0.25)" : " (NOT acceptable, > 0.25)") << "\n"
              << "PRED(0.25) = " << double(q.pred) << (q.pred >= 0.75L ? " (acceptable, >= 0.75)" : " (NOT acceptable, < 0.75)") << "\n";
}

// Steps 2-8 of the work for one normalizing transformation
NonlinearResult Analyze(const std::vector<ldouble>& x, const std::vector<ldouble>& y,
                        const Normalization& psi, ldouble significance) {
    const std::size_t n = y.size();
    NonlinearResult result;
    result.name = psi.name;

    // 2. Normalization of the empirical data
    std::cout << "\n--- 2. Normalized data ---\n";
    Matrix<ldouble> z {n, 2};
    std::vector<ldouble> zy(n);
    for (std::size_t i = 0; i < n; ++i) {
        z[i, 0] = psi.forward_x(x[i]);
        z[i, 1] = zy[i] = psi.forward_y(y[i]);
    }
    std::vector<ldouble> avg = average_of_columns(z);
    std::cout << "Average of Z1: " << double(avg[0]) << ", average of ZY: " << double(avg[1]) << "\n";
    mardia_normality_check(z, significance);

    // 3. Linear regression for the normalized data (5.3)
    std::cout << "\n--- 3. Linear regression for the normalized data ---\n";
    Matrix<ldouble> z1 {n, 1};
    for (std::size_t i = 0; i < n; ++i) z1[i, 0] = z[i, 0];
    linear_regression model = fit_linear_regression(z1, zy);
    if (!model.valid) {
        std::cout << "Matrix X^T X is singular; the parameters cannot be estimated.\n";
        return result;
    }
    ldouble b0 = model.coefficients[0], b1 = model.coefficients[1];
    std::cout << "b0 = " << double(b0) << "\n"
              << "b1 = " << double(b1) << "\n"
              << "ZY^ = " << double(b0) << (b1 < 0 ? " - " : " + ") << double(std::abs(b1)) << "*Z1\n"
              << "R^2 for the normalized data = " << double(determination_coefficient(model)) << "\n";

    // 4. Deviations between the normalized data and the linear equation
    std::cout << "\n--- 4. Deviations for the normalized data ---\n"
              << std::setw(4) << "i" << std::setw(13) << "Z1" << std::setw(13) << "ZY"
              << std::setw(13) << "ZY^" << std::setw(13) << "e" << "\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << std::setw(4) << i + 1 << std::setw(13) << double(z[i, 0]) << std::setw(13) << double(zy[i])
                  << std::setw(13) << double(model.predicted[i]) << std::setw(13) << double(model.residuals[i]) << "\n";
    }

    PrintDeviationCharacteristics(model);

    // 5. Normality of the deviations with confidence probability 0.95
    std::cout << "\n--- 5. Normality of the deviations ---\n"
              << "Average of deviations: " << double(average(model.residuals)) << "\n";
    result.residuals_normal = gaussian_normality_check(model.residuals, significance, true);
    write_histogram_svg(model.residuals, static_cast<std::size_t>(sturges_formula(n)),
                        "residuals_" + psi.tag + ".svg");

    // 6. Nonlinear equation (5.4)
    std::cout << "\n--- 6. Nonlinear regression equation ---\n";
    result.equation = psi.equation(b0, b1);
    std::cout << result.equation << "\n";
    auto predict = [&](ldouble v) { return psi.inverse_y(b0 + b1*psi.forward_x(v)); };
    std::vector<ldouble> predicted;
    std::size_t undefined = 0;
    for (auto v : x) {
        predicted.push_back(predict(v));
        if (std::isnan(predicted.back())) ++undefined;
    }
    if (undefined) std::cout << "Note: the equation is undefined for " << undefined << " point(s).\n";

    // 7. Quality of the nonlinear equation in the original scale
    std::cout << "\n--- 7. Quality of the nonlinear regression ---\n";
    result.quality = Evaluate(y, predicted);
    PrintQuality(result.quality);
    std::vector<ldouble> mre = magnitudes_of_relative_error(y, predicted);
    std::cout << std::setw(4) << "i" << std::setw(13) << "X1" << std::setw(13) << "Y"
              << std::setw(13) << "Y^" << std::setw(10) << "MRE" << "\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << std::setw(4) << i + 1 << std::setw(13) << double(x[i]) << std::setw(13) << double(y[i])
                  << std::setw(13) << double(predicted[i]) << std::setw(10) << double(mre[i])
                  << (mre[i] > 0.25L ? "  *" : "") << "\n";
    }
    std::cout << "(* marks MRE > 0.25)\n";

    // 8. The regression curve over the empirical data
    std::cout << "\n--- 8. Regression curve ---\n";
    // Only the final, compact form of the equation fits under the title
    std::string compact = "Y^ =" + result.equation.substr(result.equation.rfind(" = ") + 2);
    write_regression_curve_svg(x, y, predict, compact, "regression_" + psi.tag + ".svg");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    // 0. Read data. Default to the data file next to the sources, independent of the working directory.
    // Rows are observations; the first column is the factor X1, the last one is the dependent variable Y.
    std::string filepath = (argc > 1) ? argv[1] : std::string(DATA_DIR) + "/data.txt";
    Matrix<ldouble> data = DataReader::ReadMatrix<ldouble>(filepath);
    if (data.rows_count() < 5 || data.columns_count() < 2) {
        std::cout << "Could not read the factor and the dependent variable from " << filepath << "\n";
        return 0;
    }
    std::vector<ldouble> x = data.get_column(0);
    std::vector<ldouble> y = data.get_column(data.columns_count() - 1);
    const std::size_t n = y.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] <= 0 || y[i] <= 0) {
            std::cout << "Row " << i + 1 << " is not positive; logarithm and Box-Cox transformation are not defined.\n";
            return 0;
        }
    }
    std::cout << "Read " << n << " rows from " << filepath << "\n";

    const ldouble significance = 0.05L;   // confidence probability 0.95

    // Linear equation of work 4 for the comparison
    Matrix<ldouble> x1 {n, 1};
    for (std::size_t i = 0; i < n; ++i) x1[i, 0] = x[i];
    linear_regression linear = fit_linear_regression(x1, y);
    Quality linear_quality = Evaluate(y, linear.predicted);
    std::string linear_equation = "Y^ = " + Number(linear.coefficients[0]) + " + "
                                + Number(linear.coefficients[1]) + "*X1";
    std::cout << "\n=== Linear equation of work 4 (for the comparison) ===\n" << linear_equation << "\n";
    PrintQuality(linear_quality);
    bool linear_normal = gaussian_normality_check(linear.residuals, 0.05L, true);

    // 1. Candidate transformations, tried in this order: the decimal logarithm (5.5)-(5.6)
    // and, for the independent work, the bivariate Box-Cox transformation (3.6) with
    // parameters estimated by maximum likelihood (3.8)
    std::vector<Normalization> candidates;
    candidates.push_back({"decimal logarithm", "lg",
        [](ldouble v) { return std::log10(v); },
        [](ldouble v) { return std::log10(v); },
        [](ldouble z) { return std::pow(10.0L, z); },
        [](ldouble b0, ldouble b1) {
            return "Y^ = 10^" + Number(b0) + " * X1^" + Number(b1)
                 + " = " + Number(std::pow(10.0L, b0)) + " * X1^" + Number(b1);
        }});

    std::vector<ldouble> lambdas = fit_box_cox(data);
    const ldouble lx = lambdas[0], ly = lambdas.back();
    candidates.push_back({"Box-Cox", "boxcox",
        [lx](ldouble v) { return box_cox_transform(v, lx); },
        [ly](ldouble v) { return box_cox_transform(v, ly); },
        [ly](ldouble z) { return box_cox_inverse_transform(z, ly); },
        [lx, ly](ldouble b0, ldouble b1) {
            // (1 + lY*(b0 + b1*(X^l1 - 1)/l1))^(1/lY) = (c0 + c1*X^l1)^(1/lY)
            ldouble c0 = 1 + ly*(b0 - b1/lx), c1 = ly*b1/lx;
            return "Y^ = (1 + " + Number(ly) + "*(" + Number(b0) + " + " + Number(b1) + "*(X1^" + Number(lx)
                 + " - 1)/" + Number(lx) + "))^(1/" + Number(ly) + ") = (" + Number(c0) + (c1 < 0 ? " - " : " + ")
                 + Number(std::abs(c1)) + "*X1^" + Number(lx) + ")^" + Number(1/ly);
        }});
    std::cout << "Box-Cox parameters (ML estimate of (3.8)): lambda1 = " << double(lx)
              << ", lambdaY = " << double(ly) << "\n";

    std::vector<NonlinearResult> results;
    std::size_t accepted = candidates.size();
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        std::cout << "\n=== 1. Transformation: " << candidates[c].name << " ===\n";
        results.push_back(Analyze(x, y, candidates[c], significance));
        if (accepted == candidates.size()) {
            if (results.back().residuals_normal) {
                accepted = c;
            } else if (c + 1 < candidates.size()) {
                std::cout << "Deviations are not normal: back to step 1 with the next transformation.\n";
            }
        }
    }

    // 9. Comparison with the linear equation of work 4 and between the transformations
    std::cout << "\n=== 9. Comparison ===\n"
              << std::left << std::setw(48) << "Equation" << std::right
              << std::setw(10) << "R^2" << std::setw(10) << "MMRE" << std::setw(12) << "PRED(0.25)"
              << std::setw(16) << "Normal dev." << "\n";
    std::cout << std::left << std::setw(48) << "linear (work 4)" << std::right << std::fixed << std::setprecision(4)
              << std::setw(10) << double(linear_quality.r2) << std::setw(10) << double(linear_quality.mmre)
              << std::setw(12) << double(linear_quality.pred) << std::setw(16) << (linear_normal ? "yes" : "no") << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(48) << ("nonlinear, " + r.name) << std::right
                  << std::setw(10) << double(r.quality.r2) << std::setw(10) << double(r.quality.mmre)
                  << std::setw(12) << double(r.quality.pred) << std::setw(16) << (r.residuals_normal ? "yes" : "no") << "\n";
    }
    std::cout << std::defaultfloat << std::setprecision(6);

    std::cout << "\n=== Conclusions ===\n"
              << "Linear equation of work 4: " << linear_equation << "\n";
    for (const auto& r : results) std::cout << "Nonlinear, " << r.name << ": " << r.equation << "\n";
    if (accepted < results.size()) {
        const auto& best = results[accepted];
        std::cout << "Deviations for the transformation " << best.name << " are normal with confidence probability "
                  << double(1 - significance) << ", so the nonlinear equation is theoretically justified.\n";
        bool better = best.quality.r2 > linear_quality.r2 && best.quality.mmre < linear_quality.mmre
                   && best.quality.pred > linear_quality.pred;
        std::cout << "Compared to the linear equation it is " << (better ? "better by all three indicators" : "not better by all indicators")
                  << "; acceptable by: " << (best.quality.r2 > 0.75L ? "R^2 " : "") << (best.quality.mmre <= 0.25L ? "MMRE " : "")
                  << (best.quality.pred >= 0.75L ? "PRED(0.25)" : "")
                  << (best.quality.r2 > 0.75L || best.quality.mmre <= 0.25L || best.quality.pred >= 0.75L ? "" : "none") << "\n";
    } else {
        std::cout << "None of the transformations made the deviations normal; another transformation is needed.\n";
    }
}
