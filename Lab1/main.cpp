//Containers
#include <algorithm>
#include <cstddef>
#include <vector>
#include <string>

//File reading and logging
#include <iostream>

//Math
#include <cmath>
#include "formulas.hpp"
#include "analysis.hpp"
#include "data_reader.hpp"

void PrintValues(const std::string& caption, const std::vector<ldouble>& data);
bool AnalyzeSample(const std::vector<ldouble>& data, ldouble significance, int pass);

int main (int argc, char** argv) {
    // 0. Read numbers AND sort data
    // Default to the data file next to the sources, independent of the working directory
    std::string arg = (argc > 1) ? argv[1] : std::string(DATA_DIR) + "/data.txt";
    auto data = DataReader::ReadVector<ldouble>(arg);
    if (data.size() < 3) {
        std::cout << "Not enough values to analyze.\n";
        return 0;
    }
    std::sort(data.begin(), data.end());

    ldouble significance = 0.05;

    const std::size_t initial_size = data.size();

    PrintValues("Input values:", data);

    // Precalculate values, check normality, find trusted ranges
    bool isNormal = AnalyzeSample(data, significance, 0);

    // Censor the sample and repeat the analysis until nothing is dropped
    for (int pass = 1; ; ++pass) {
        std::cout << "\n=== Censoring, pass " << pass << " ===\n";

        std::vector<ldouble> censored = censor_data(data, isNormal, significance);

        std::size_t dropped = data.size() - censored.size();
        std::cout << "Dropped values: " << dropped << "\n";

        if (dropped == 0) {
            std::cout << "No outliers left; analysis is finished.\n";
            break;
        }
        if (censored.size() < 3) {
            std::cout << "Too few values left after censoring; stopping.\n";
            break;
        }

        data = std::move(censored);
        PrintValues("Censored values:", data);

        std::cout << "\n=== Repeated analysis of the censored sample ===\n";
        isNormal = AnalyzeSample(data, significance, pass);
    }

    // 7. Conclusions.
    std::cout << "\n=== Conclusions ===\n"
    << "Values kept: " << data.size() << " of " << initial_size
    << " (" << (initial_size - data.size()) << " discarded as outliers)\n"
    << "Distribution law: " << (isNormal ? "may be considered normal"
                                         : "cannot be considered normal")
    << " at the " << double(100*significance) << "% significance level\n";
}

bool AnalyzeSample(const std::vector<ldouble>& data, ldouble significance, int pass) {
    ldouble len = data.size();
    ldouble avg = average(data);
    ldouble variance = unbiased_point_estimate_of_variance(avg, data);
    ldouble deviation = std::sqrt(variance);
    ullong sections_count = sturges_formula(len);

    ldouble t_quantile = student_quantile(significance/2, len-1);
    std::pair<ldouble, ldouble> trusted_avg_range = trusted_average_range(len, avg, deviation, t_quantile);
    std::pair<ldouble, ldouble> trusted_dev_range = trusted_deviation_range(len, deviation, significance);

    std::cout << "Precalculation data: \n"
    << "Average: " << double(avg) << '\n'
    << "Variance: " << double(variance) << '\n'
    << "Deviation: " << double(deviation) << '\n'
    << "Sections count: " << sections_count << '\n';

    write_histogram_svg(data, sections_count, "histogram_pass" + std::to_string(pass) + ".svg");

    bool isNormal = gaussian_normality_check(data, significance);
    if (isNormal) {
        std::cout
        << "Trusted average range: [" << double(trusted_avg_range.first) << "; " << double(trusted_avg_range.second) << "]\n"
        << "Trusted deviation range: [" << double(trusted_dev_range.first) << "; " << double(trusted_dev_range.second) << "]\n";
    }

    return isNormal;
}

void PrintValues(const std::string& caption, const std::vector<ldouble>& data) {
    std::cout << caption;
    for (auto t : data) {
        std::cout << " " << double(t);
    }
    std::cout << std::endl;
}
