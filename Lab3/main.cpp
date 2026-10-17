//Containers
#include <cstddef>
#include <string>
#include <vector>

//File reading and logging
#include <iostream>

//Math
#include "formulas.hpp"
#include "analysis.hpp"
#include "matrix.hpp"
#include "data_reader.hpp"

namespace {

struct AnalysisResult {
    Matrix<ldouble> cleaned;
    std::vector<std::size_t> removed;   // indices of the discarded rows in the input file
    bool normal_initially {false};
};

Matrix<ldouble> without_row(const Matrix<ldouble>& data, std::size_t skipped) {
    Matrix<ldouble> result {data.rows_count() - 1, data.columns_count()};
    for (std::size_t i = 0, out = 0; i < data.rows_count(); ++i) {
        if (i == skipped) continue;
        for (std::size_t j = 0; j < data.columns_count(); ++j)
            result[out, j] = data[i, j];
        ++out;
    }
    return result;
}

void PrintVector(const std::string& caption, const std::vector<ldouble>& values) {
    std::cout << caption;
    for (auto v : values) std::cout << ' ' << double(v);
    std::cout << "\n";
}

// Steps 1-5 of the work for one significance level
AnalysisResult Analyze(Matrix<ldouble> data, ldouble significance, const std::string& tag) {
    AnalysisResult result;
    std::vector<std::size_t> ids(data.rows_count());
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i;

    // 1-2. Vector of means and covariance matrix
    PrintVector("Average:", average_of_columns(data));
    std::cout << "Covariance matrix (biased): " << biased_covariance_matrix(data) << "\n"
              << "Generalized variance (det S): " << double(biased_covariance_matrix(data).determinant()) << "\n\n";

    // 3. Deviation of the distribution from the normal law
    std::cout << "--- Mardia test of the input data ---\n";
    result.normal_initially = mardia_normality_check(data, significance);

    // 4. Confidence ellipsoid of the means
    std::cout << "\n--- Input data ---\n";
    print_confidence_ellipsoid(data, significance);

    // 5. Outliers: prediction ellipsoid of the normalized data, one point per pass
    const std::size_t k = data.columns_count();
    for (int pass = 1; data.rows_count() > k + 2; ++pass) {
        std::cout << "\n=== Outliers, pass " << pass << " (N = " << data.rows_count() << ") ===\n";

        Matrix<ldouble> working = data;
        bool normal = (pass == 1) ? result.normal_initially : mardia_normality_check(data, significance);
        if (!normal) {
            std::vector<ldouble> lambdas = fit_box_cox(data);
            if (lambdas.empty()) {
                std::cout << "Data are not positive, Box-Cox transformation is not defined; "
                             "the data are used as they are.\n";
            } else {
                working = apply_box_cox(data, lambdas);
                PrintVector("Box-Cox parameters (ML estimate of (3.8)):", lambdas);
                std::cout << "Log-likelihood: " << double(box_cox_log_likelihood(data, lambdas)) << "\n"
                          << "Mardia test of the normalized data:\n";
                if (!mardia_normality_check(working, significance)) {
                    std::cout << "Note: the normalized data still deviate from the normal law; "
                                 "the multivariate Johnson transformation is recommended instead.\n";
                }
            }
        }

        std::vector<ldouble> d2 = squared_mahalanobis_distances(working);
        std::size_t worst = 0;
        for (std::size_t i = 1; i < d2.size(); ++i)
            if (d2[i] > d2[worst]) worst = i;

        ldouble border = prediction_ellipsoid_border(data.rows_count(), k, significance);
        bool outlier = d2[worst] > border;
        std::cout << "Prediction ellipsoid border: " << double(border) << "\n"
                  << "Largest squared Mahalanobis distance: " << double(d2[worst])
                  << " (row " << ids[worst] + 1 << ")\n";

        write_scatter_svg(working, outlier ? worst : data.rows_count(), significance,
                          std::string("Prediction ellipse of the ") + (normal ? "" : "normalized ") + "data",
                          "scatter_" + tag + "_pass" + std::to_string(pass) + ".svg");

        if (!outlier) {
            std::cout << "No outliers left.\n";
            break;
        }
        std::cout << "Row " << ids[worst] + 1 << " is an outlier: ";
        for (std::size_t j = 0; j < k; ++j) std::cout << double(data[worst, j]) << ' ';
        std::cout << "-> discarded\n";

        result.removed.push_back(ids[worst]);
        ids.erase(ids.begin() + worst);
        data = without_row(data, worst);
    }

    std::cout << "\n--- Cleaned data (N = " << data.rows_count() << ") ---\n";
    print_confidence_ellipsoid(data, significance);
    result.cleaned = data;
    return result;
}

void PrintRemoved(const AnalysisResult& r) {
    if (r.removed.empty()) { std::cout << "none"; return; }
    for (auto i : r.removed) std::cout << "row " << i + 1 << ' ';
}

} // namespace

int main(int argc, char** argv) {
    // 0. Read data. Default to the data file next to the sources, independent of the working directory.
    // Rows are observations, columns are variables; decimal commas are normalized by DataReader.
    std::string filepath = (argc > 1) ? argv[1] : std::string(DATA_DIR) + "/data.txt";
    Matrix<ldouble> data = DataReader::ReadMatrix<ldouble>(filepath);
    if (data.rows_count() == 0) {
        std::cout << "Could not read any data from " << filepath << "\n";
        return 0;
    }
    if (data.rows_count() <= data.columns_count() + 2) {
        std::cout << "Not enough observations for the analysis.\n";
        return 0;
    }

    std::cout << "Read " << data.rows_count() << " rows x " << data.columns_count()
              << " columns from " << filepath << "\n";

    const ldouble main_significance = 0.005L;
    const ldouble extra_significance = 0.05L;

    std::cout << "\n=== Significance level " << double(100*main_significance) << "% ===\n";
    AnalysisResult main_result = Analyze(data, main_significance, "a0005");

    // Independent work: the same analysis at the 5% level
    std::cout << "\n\n=== Significance level " << double(100*extra_significance) << "% ===\n";
    AnalysisResult extra_result = Analyze(data, extra_significance, "a005");

    // 6. Conclusions
    std::cout << "\n=== Conclusions ===\n"
    << "At the " << double(100*main_significance) << "% significance level the input data "
    << (main_result.normal_initially ? "may be considered" : "cannot be considered") << " multivariate normal; discarded: ";
    PrintRemoved(main_result);
    std::cout << " (" << main_result.cleaned.rows_count() << " of " << data.rows_count() << " kept)\n"
    << "At the " << double(100*extra_significance) << "% significance level the input data "
    << (extra_result.normal_initially ? "may be considered" : "cannot be considered") << " multivariate normal; discarded: ";
    PrintRemoved(extra_result);
    std::cout << " (" << extra_result.cleaned.rows_count() << " of " << data.rows_count() << " kept)\n";
    if (extra_result.removed.size() > main_result.removed.size()) {
        std::cout << "The stricter 5% level flags more points: its prediction ellipsoid is smaller, "
                     "while the 0.5% level is more tolerant to deviations.\n";
    }
    std::cout << "Note: insignificance of the statistics does not necessarily mean normality.\n";
}
