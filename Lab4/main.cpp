//Containers
#include <cstddef>
#include <string>
#include <vector>

//File reading and logging
#include <cmath>
#include <iomanip>
#include <iostream>

//Math
#include "formulas.hpp"
#include "analysis.hpp"
#include "matrix.hpp"
#include "data_reader.hpp"

namespace {

// VIF above this value signals multicollinearity; from 1 to 5 there is none
constexpr ldouble vif_border = 10;
constexpr ldouble vif_safe = 5;

Matrix<ldouble> without_column(const Matrix<ldouble>& data, std::size_t skipped) {
    Matrix<ldouble> result {data.rows_count(), data.columns_count() - 1};
    for (std::size_t j = 0, out = 0; j < data.columns_count(); ++j) {
        if (j == skipped) continue;
        for (std::size_t i = 0; i < data.rows_count(); ++i)
            result[i, out] = data[i, j];
        ++out;
    }
    return result;
}

// Step 1: discards the factor with the largest VIF while it exceeds the border
Matrix<ldouble> drop_multicollinear_factors(Matrix<ldouble> x, std::vector<std::size_t>& names) {
    while (x.columns_count() > 1) {
        std::vector<ldouble> vif = variance_inflation_factors(x);
        std::size_t worst = 0;
        std::cout << "VIFs:";
        for (std::size_t j = 0; j < vif.size(); ++j) {
            std::cout << " X" << names[j] << '=' << double(vif[j]);
            if (vif[j] > vif[worst]) worst = j;
        }
        std::cout << "\n";

        if (vif[worst] <= vif_border) {
            std::cout << (vif[worst] <= vif_safe ? "All VIFs are within [1; 5]: no multicollinearity.\n"
                                                 : "VIFs do not exceed 10: multicollinearity is moderate, "
                                                   "all factors are kept.\n");
            break;
        }
        std::cout << "X" << names[worst] << " has VIF > " << double(vif_border)
                  << ": multicollinearity, the factor is discarded.\n";
        x = without_column(x, worst);
        names.erase(names.begin() + worst);
    }
    return x;
}

void PrintEquation(const linear_regression& model, const std::vector<std::size_t>& names) {
    std::cout << "Y^ = " << double(model.coefficients[0]);
    for (std::size_t j = 1; j < model.coefficients.size(); ++j) {
        ldouble b = model.coefficients[j];
        std::cout << (b < 0 ? " - " : " + ") << double(std::abs(b)) << "*X" << names[j - 1];
    }
    std::cout << "\n";
}

// Independent work: T-tests (4.12)-(4.13) of every parameter and the ANOVA F-test (4.14)
bool SignificanceTests(const linear_regression& model, std::size_t k, ldouble significance) {
    const ldouble n = static_cast<ldouble>(model.residuals.size()), kk = static_cast<ldouble>(k);
    const ldouble freedom = n - kk - 1;
    ldouble t_border = student_quantile(significance/2, freedom);

    std::cout << "Student quantile t(" << double(significance/2) << ", " << double(freedom) << "): "
              << double(t_border) << "\n";
    bool slopes_significant = true;
    for (std::size_t j = 0; j < model.coefficients.size(); ++j) {
        ldouble t = model.coefficients[j]/model.standard_errors[j];
        bool rejected = std::abs(t) > t_border;
        std::cout << "  b" << j << ": se = " << double(model.standard_errors[j]) << ", T0 = " << double(t)
                  << " -> H0: b" << j << " = 0 " << (rejected ? "REJECTED (significant)" : "NOT REJECTED") << "\n";
        if (j > 0 && !rejected) slopes_significant = false;
    }

    ldouble msr = model.ssr/kk, mse = model.sse/freedom;
    ldouble f = msr/mse;
    ldouble f_border = fisher_quantile(significance, kk, freedom);
    bool regression_significant = f > f_border;
    std::cout << "SSR = " << double(model.ssr) << ", SSE = " << double(model.sse)
              << ", SST = " << double(model.sst) << "\n"
              << "MSR = " << double(msr) << ", MSE = " << double(mse) << "\n"
              << "F0 = " << double(f) << ", F(" << double(significance) << ", " << double(kk) << ", "
              << double(freedom) << ") = " << double(f_border) << " -> H0: b1 = ... = bk = 0 "
              << (regression_significant ? "REJECTED: the linear dependence exists" : "NOT REJECTED") << "\n";
    return regression_significant && slopes_significant;
}

} // namespace

int main(int argc, char** argv) {
    // 0. Read data. Default to the data file next to the sources, independent of the working directory.
    // Rows are observations; the last column is the dependent variable Y, the others are the factors X.
    std::string filepath = (argc > 1) ? argv[1] : std::string(DATA_DIR) + "/data.txt";
    Matrix<ldouble> data = DataReader::ReadMatrix<ldouble>(filepath);
    if (data.rows_count() == 0 || data.columns_count() < 2) {
        std::cout << "Could not read the factors and the dependent variable from " << filepath << "\n";
        return 0;
    }

    std::cout << "Read " << data.rows_count() << " rows x " << data.columns_count()
              << " columns from " << filepath << "\n";

    std::vector<ldouble> y = data.get_column(data.columns_count() - 1);
    Matrix<ldouble> x = without_column(data, data.columns_count() - 1);
    std::vector<std::size_t> names;
    for (std::size_t j = 0; j < x.columns_count(); ++j) names.push_back(j + 1);

    if (data.rows_count() <= x.columns_count() + 2) {
        std::cout << "Not enough observations for the analysis.\n";
        return 0;
    }

    const ldouble significance = 0.05L;   // confidence probability 0.95
    const ldouble mre_level = 0.25L;

    // 1. Multicollinearity of the factors (skipped for one factor)
    std::cout << "\n=== 1. Multicollinearity ===\n";
    if (x.columns_count() > 1) {
        x = drop_multicollinear_factors(x, names);
    } else {
        std::cout << "One-factor regression: the step is not performed.\n";
    }
    const std::size_t k = x.columns_count();

    // 2. Least squares estimates of the parameters
    std::cout << "\n=== 2. Parameters of the regression equation ===\n";
    linear_regression model = fit_linear_regression(x, y);
    if (!model.valid) {
        std::cout << "Matrix X^T X is singular; the parameters cannot be estimated.\n";
        return 0;
    }
    for (std::size_t j = 0; j < model.coefficients.size(); ++j)
        std::cout << "b" << j << " = " << double(model.coefficients[j]) << "\n";
    PrintEquation(model, names);

    // 3. Quality of the prediction
    std::cout << "\n=== 3. Quality of the regression ===\n";
    ldouble r2 = determination_coefficient(model);
    ldouble r2_adjusted = adjusted_determination_coefficient(model, k);
    std::vector<ldouble> mre = magnitudes_of_relative_error(y, model.predicted);
    ldouble mmre = mean_magnitude_of_relative_error(mre);
    ldouble pred = percentage_of_prediction(mre, mre_level);
    bool r2_ok = r2 > 0.75L, mmre_ok = mmre <= mre_level, pred_ok = pred >= 0.75L;
    std::cout << "R^2 = " << double(r2) << (r2_ok ? " (acceptable, > 0.75)" : " (NOT acceptable, <= 0.75)") << "\n"
              << "Adjusted R^2 = " << double(r2_adjusted) << "\n"
              << "MMRE = " << double(mmre) << (mmre_ok ? " (acceptable, <= 0.25)" : " (NOT acceptable, > 0.25)") << "\n"
              << "PRED(0.25) = " << double(pred) << (pred_ok ? " (acceptable, >= 0.75)" : " (NOT acceptable, < 0.75)") << "\n";

    // 4. The regression line over the empirical data
    if (k == 1) {
        std::cout << "\n=== 4. Regression line ===\n";
        write_regression_svg(x.get_column(0), y, model.coefficients[0], model.coefficients[1],
                             "regression_line.svg");
    }

    // 5. Deviations between the empirical data and the regression line (4.16)
    std::cout << "\n=== 5. Deviations ===\n"
              << std::setw(4) << "i";
    for (auto name : names) std::cout << std::setw(13) << ("X" + std::to_string(name));
    std::cout << std::setw(13) << "Y" << std::setw(13) << "Y^" << std::setw(13) << "e" << std::setw(10) << "MRE" << "\n";
    for (std::size_t i = 0; i < y.size(); ++i) {
        std::cout << std::setw(4) << i + 1;
        for (std::size_t j = 0; j < k; ++j) std::cout << std::setw(13) << double(x[i, j]);
        std::cout << std::setw(13) << double(y[i]) << std::setw(13) << double(model.predicted[i])
                  << std::setw(13) << double(model.residuals[i]) << std::setw(10) << double(mre[i])
                  << (mre[i] > mre_level ? "  *" : "") << "\n";
    }
    std::cout << "(* marks MRE > 0.25)\n";

    // 6. Normality of the deviations with confidence probability 0.95
    std::cout << "\n=== 6. Normality of the deviations ===\n";
    std::cout << "Average of deviations: " << double(average(model.residuals)) << "\n";
    bool residuals_normal = gaussian_normality_check(model.residuals, significance);
    write_histogram_svg(model.residuals, static_cast<std::size_t>(sturges_formula(model.residuals.size())),
                        "residuals_histogram.svg");

    // Independent work: significance of the regression
    std::cout << "\n=== Significance of the regression (T-test and F-test) ===\n";
    bool significant = SignificanceTests(model, k, significance);

    // 7. Conclusions
    std::cout << "\n=== 7. Conclusions ===\n";
    PrintEquation(model, names);
    std::cout << "The equation explains " << double(100*r2) << "% of the variability of Y; "
              << "MMRE = " << double(mmre) << ", PRED(0.25) = " << double(pred) << ".\n";
    if (r2_ok && mmre_ok && pred_ok) {
        std::cout << "All three quality indicators are acceptable: the equation estimates the metric with good accuracy.\n";
    } else {
        std::cout << "Not acceptable:"
                  << (r2_ok ? "" : " R^2") << (mmre_ok ? "" : " MMRE") << (pred_ok ? "" : " PRED(0.25)")
                  << ". The accuracy of the estimation is insufficient; another (e.g. nonlinear or "
                     "normalized) model should be considered.\n";
    }
    std::cout << "The regression is " << (significant ? "" : "NOT ") << "significant at the "
              << double(100*significance) << "% level by the T- and F-tests.\n"
              << "Deviations " << (residuals_normal ? "may be considered" : "cannot be considered")
              << " normally distributed with confidence probability " << double(1 - significance) << ": "
              << (residuals_normal ? "application of the linear regression is theoretically justified.\n"
                                   : "application of the linear regression is NOT theoretically justified.\n");
}
