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

int main(int argc, char** argv) {
    // 0. Read data. Default to the data file next to the sources, independent of the working directory.
    // The file is organized column-wise, with decimal commas in some cells (e.g. "2496 6,4");
    // DataReader normalizes those before parsing.
    std::string filepath = (argc > 1) ? argv[1] : std::string(DATA_DIR) + "/data.txt";
    Matrix<ldouble> data = DataReader::ReadMatrix<ldouble>(filepath);
    if (data.rows_count() == 0) {
        std::cout << "Could not read any data from " << filepath << "\n";
        return 0;
    }

    std::cout << "Read " << data.rows_count() << " rows x " << data.columns_count()
              << " columns from " << filepath << "\n\n";

    // Precalculate values
    std::vector<ldouble> average = average_of_columns(data);
    std::cout << "Average:";
    for (auto a : average) std::cout << ' ' << double(a);
    std::cout << "\nCovariance matrix (biased): " << biased_covariance_matrix(data) << "\n\n";

    // 1-3. Mardia skewness and kurtosis, test statistics, check of the deviation from normal law
    const ldouble main_significance = 0.005L;
    const ldouble extra_significance = 0.05L;

    std::cout << "=== Significance level " << double(100*main_significance) << "% ===\n";
    bool normal_main = mardia_normality_check(data, main_significance);

    // Independent work: the same check at the 5% level
    std::cout << "\n=== Significance level " << double(100*extra_significance) << "% ===\n";
    bool normal_extra = mardia_normality_check(data, extra_significance);

    // 4. Conclusions
    std::cout << "\n=== Conclusions ===\n"
    << "At the " << double(100*main_significance) << "% significance level the data "
    << (normal_main ? "may be considered" : "cannot be considered") << " multivariate normal\n"
    << "At the " << double(100*extra_significance) << "% significance level the data "
    << (normal_extra ? "may be considered" : "cannot be considered") << " multivariate normal\n";
    if (normal_main != normal_extra) {
        std::cout << "The verdict depends on the significance level: the stricter 5% level "
                     "rejects more readily, the 0.5% level is more tolerant to deviations.\n";
    }
    std::cout << "Note: insignificance of the statistics does not necessarily mean normality.\n";
}
