#include "analysis.hpp"
#include "formulas.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Rounds a range step up to 1, 2, 5 or 10 times a power of ten.
ldouble nice_step(ldouble range, int target_ticks) {
    if (range <= 0) return 1;
    ldouble raw = range/target_ticks;
    ldouble magnitude = std::pow(10.0L, std::floor(std::log10(raw)));
    ldouble normalized = raw/magnitude;
    ldouble step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
    return step*magnitude;
}

std::string number_to_text(ldouble value, int decimals) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << double(value);
    return out.str();
}

// A bar anchored to the baseline, with only its top corners rounded.
std::string bar_path(ldouble x, ldouble y, ldouble width, ldouble baseline) {
    ldouble height = baseline - y;
    ldouble radius = std::min({4.0L, width/2, height});
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << "M" << double(x) << " " << double(baseline)
        << "L" << double(x) << " " << double(y + radius)
        << "Q" << double(x) << " " << double(y) << " " << double(x + radius) << " " << double(y)
        << "L" << double(x + width - radius) << " " << double(y)
        << "Q" << double(x + width) << " " << double(y) << " "
               << double(x + width) << " " << double(y + radius)
        << "L" << double(x + width) << " " << double(baseline) << "Z";
    return out.str();
}

} // namespace

void write_histogram_svg(std::vector<ldouble> data, std::size_t sections_count,
                         const std::string& filename) {
    if (data.size() < 2 || sections_count == 0) return;

    std::sort(data.begin(), data.end());

    ldouble avg = average(data);
    ldouble rsmd = std::sqrt(unbiased_point_estimate_of_variance(avg, data));
    ldouble min = data.front(), max = data.back();
    ldouble delta = (max - min)/static_cast<ldouble>(sections_count);
    if (delta == 0 || rsmd == 0) {
        std::cout << "All values are identical; histogram is not drawn.\n";
        return;
    }

    std::vector<std::size_t> counts(sections_count, 0);
    for (auto v : data) {
        std::size_t idx = static_cast<std::size_t>((v - min)/delta);
        if (idx >= sections_count) idx = sections_count - 1;   // catches v == max
        ++counts[idx];
    }

    // The density is scaled by N*delta so that it shares the units of the bars:
    // its height is the frequency the normal law expects at that point.
    auto scaled_density = [&](ldouble x) {
        return gaussian_distribution_density(x, avg, rsmd)*data.size()*delta;
    };

    ldouble peak = 0;
    for (auto c : counts) peak = std::max(peak, static_cast<ldouble>(c));
    constexpr int curve_points = 240;
    for (int i = 0; i <= curve_points; ++i) {
        ldouble x = min + (max - min)*i/curve_points;
        peak = std::max(peak, scaled_density(x));
    }

    ldouble y_step = nice_step(peak, 5);
    ldouble y_max = std::ceil(peak/y_step)*y_step;

    constexpr ldouble width = 880, height = 520;
    constexpr ldouble pad_left = 68, pad_right = 28, pad_top = 64, pad_bottom = 72;
    const ldouble plot_width = width - pad_left - pad_right;
    const ldouble plot_height = height - pad_top - pad_bottom;
    const ldouble baseline = pad_top + plot_height;

    auto to_x = [&](ldouble v) { return pad_left + (v - min)/(max - min)*plot_width; };
    auto to_y = [&](ldouble v) { return baseline - v/y_max*plot_height; };

    std::ofstream svg {filename};
    if (!svg.is_open()) {
        std::cout << "Could not write the histogram to " << filename << "\n";
        return;
    }
    svg << std::fixed << std::setprecision(2);

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << double(width)
        << "\" height=\"" << double(height) << "\" viewBox=\"0 0 " << double(width)
        << " " << double(height) << "\" font-family=\"Inter, Segoe UI, Helvetica, Arial, sans-serif\">\n";

    svg << "<style>\n"
           "  .surface{fill:#fcfcfb}\n"
           "  .ink{fill:#0b0b0b}\n"
           "  .ink2{fill:#52514e}\n"
           "  .muted{fill:#898781}\n"
           "  .grid{stroke:#e1e0d9;stroke-width:1}\n"
           "  .axis{stroke:#c3c2b7;stroke-width:1}\n"
           "  .bar{fill:#2a78d6}\n"
           "  .curve{stroke:#eb6834;stroke-width:2;fill:none;"
           "stroke-linejoin:round;stroke-linecap:round}\n"
           "  @media (prefers-color-scheme: dark){\n"
           "    .surface{fill:#1a1a19} .ink{fill:#ffffff} .ink2{fill:#c3c2b7}\n"
           "    .grid{stroke:#2c2c2a} .axis{stroke:#383835}\n"
           "    .bar{fill:#3987e5} .curve{stroke:#d95926}\n"
           "  }\n"
           "</style>\n";

    svg << "<rect class=\"surface\" width=\"" << double(width) << "\" height=\"" << double(height) << "\"/>\n";

    svg << "<text class=\"ink\" x=\"" << double(pad_left) << "\" y=\"30\" font-size=\"17\" "
           "font-weight=\"600\">Histogram and fitted normal density</text>\n";
    svg << "<text class=\"ink2\" x=\"" << double(pad_left) << "\" y=\"50\" font-size=\"12.5\">N = "
        << data.size() << ",  mean = " << number_to_text(avg, 3)
        << ",  deviation = " << number_to_text(rsmd, 3)
        << ",  intervals = " << sections_count << "</text>\n";

    // Horizontal gridlines and the frequency scale
    for (ldouble v = 0; v <= y_max + y_step/2; v += y_step) {
        ldouble y = to_y(v);
        svg << "<line class=\"" << (v == 0 ? "axis" : "grid") << "\" x1=\"" << double(pad_left)
            << "\" y1=\"" << double(y) << "\" x2=\"" << double(pad_left + plot_width)
            << "\" y2=\"" << double(y) << "\"/>\n";
        svg << "<text class=\"muted\" x=\"" << double(pad_left - 10) << "\" y=\"" << double(y + 4)
            << "\" font-size=\"11.5\" text-anchor=\"end\">" << number_to_text(v, 0) << "</text>\n";
    }

    // Bars, separated by a 2px gap so neighbouring fills never touch
    ldouble slot = plot_width/static_cast<ldouble>(sections_count);
    for (std::size_t j = 0; j < sections_count; ++j) {
        if (counts[j] == 0) continue;
        ldouble x = pad_left + slot*j + 1;
        ldouble bar_width = slot - 2;
        ldouble y = to_y(static_cast<ldouble>(counts[j]));
        svg << "<path class=\"bar\" d=\"" << bar_path(x, y, bar_width, baseline) << "\"/>\n";
        svg << "<text class=\"ink2\" x=\"" << double(x + bar_width/2) << "\" y=\"" << double(y - 7)
            << "\" font-size=\"11.5\" text-anchor=\"middle\">" << counts[j] << "</text>\n";
    }

    // The density curve
    svg << "<path class=\"curve\" d=\"";
    for (int i = 0; i <= curve_points; ++i) {
        ldouble x = min + (max - min)*i/curve_points;
        svg << (i == 0 ? "M" : "L") << double(to_x(x)) << " " << double(to_y(scaled_density(x)));
    }
    svg << "\"/>\n";

    // Interval boundaries on the value axis, thinned out so labels never collide
    std::size_t label_step = (sections_count > 10) ? (sections_count/10 + 1) : 1;
    for (std::size_t j = 0; j <= sections_count; j += label_step) {
        ldouble value = min + delta*j;
        ldouble x = pad_left + slot*j;
        svg << "<line class=\"axis\" x1=\"" << double(x) << "\" y1=\"" << double(baseline)
            << "\" x2=\"" << double(x) << "\" y2=\"" << double(baseline + 5) << "\"/>\n";
        svg << "<text class=\"muted\" x=\"" << double(x) << "\" y=\"" << double(baseline + 20)
            << "\" font-size=\"11\" text-anchor=\"middle\">" << number_to_text(value, 1) << "</text>\n";
    }

    svg << "<text class=\"ink2\" x=\"" << double(pad_left + plot_width/2) << "\" y=\""
        << double(height - 26) << "\" font-size=\"12\" text-anchor=\"middle\">Value</text>\n";
    svg << "<text class=\"ink2\" transform=\"translate(18," << double(pad_top + plot_height/2)
        << ") rotate(-90)\" font-size=\"12\" text-anchor=\"middle\">Frequency</text>\n";

    // Legend
    ldouble legend_x = pad_left + plot_width - 285, legend_y = pad_top - 22;
    svg << "<rect class=\"bar\" x=\"" << double(legend_x) << "\" y=\"" << double(legend_y - 8)
        << "\" width=\"11\" height=\"11\" rx=\"2\"/>\n";
    svg << "<text class=\"ink2\" x=\"" << double(legend_x + 17) << "\" y=\"" << double(legend_y + 1)
        << "\" font-size=\"12\">Observed frequency</text>\n";
    svg << "<line class=\"curve\" x1=\"" << double(legend_x + 150) << "\" y1=\"" << double(legend_y - 3)
        << "\" x2=\"" << double(legend_x + 168) << "\" y2=\"" << double(legend_y - 3) << "\"/>\n";
    svg << "<text class=\"ink2\" x=\"" << double(legend_x + 174) << "\" y=\"" << double(legend_y + 1)
        << "\" font-size=\"12\">Normal density</text>\n";

    svg << "</svg>\n";
    svg.close();

    std::cout << "Histogram written to " << filename << "\n";
}

bool gaussian_normality_check(std::vector<ldouble> data, ldouble significance) {
    bool isNormal {false};

    std::sort(data.begin(), data.end());

    if (data.size() > 30) {
        ldouble avg = average(data);
        ldouble rsmd = std::sqrt(unbiased_point_estimate_of_variance(avg, data));
        if (rsmd == 0) {
            std::cout << "All values are identical; normality check is not applicable.\n";
            return false;
        }

        std::vector<std::pair<ullong, ldouble>> frequency_chance_pairs {};

        std::size_t sections_count = static_cast<std::size_t>(sturges_formula(data.size()));
        if (sections_count < 4) sections_count = 4;
        ldouble freedom_degree = static_cast<ldouble>(sections_count) - 3;

        ldouble min = data.front(), max = data.back();
        ldouble delta = (max - min)/static_cast<ldouble>(sections_count);

        for (std::size_t j = 0; j < sections_count; ++j) {
            ldouble low = min + delta*j, high = min + delta*(j+1);
            ullong counter = 0;
            for (auto v : data) {
                if (v >= low) {
                    if (v < high || (j == sections_count-1)) {
                        ++counter;
                    } else {
                        break;
                    }
                }
            }
            // The outer intervals are stretched to infinity so that the
            // probabilities of all intervals add up to exactly one.
            ldouble low_cdf  = (j == 0)
                             ? 0.0L
                             : theoretical_cumulative_distribution_function(low, avg, rsmd);
            ldouble high_cdf = (j == sections_count - 1)
                             ? 1.0L
                             : theoretical_cumulative_distribution_function(high, avg, rsmd);
            ldouble chance = high_cdf - low_cdf;

            frequency_chance_pairs.push_back({counter, chance});
        }

        ldouble pc = pearson_criteria(data.size(), frequency_chance_pairs);
        ldouble pcc = pearson_criteria_critical(significance, freedom_degree);

        std::cout << "Checking normality hypothesis using Pearson Criteria:\n"
        << "Size of sample (N): " << data.size() << "\n"
        << "Sections count (m): " << sections_count << "\n"
        << "Freedom degree (v): " << freedom_degree << "\n"
        << "Pearson criteria value: " << double(pc) << " \n"
        << "Pearson criteria border: " << double(pcc) << "\n";

        for (std::size_t j = 0; j < frequency_chance_pairs.size(); ++j) {
            ldouble expected = data.size()*frequency_chance_pairs[j].second;
            if (expected < 5) {
                std::cout << "  Warning: interval " << j << " has expected count "
                          << double(expected) << " < 5; the result may be unreliable.\n";
            }
        }

        if (pc > pcc) {
            isNormal = false;
            std::cout << "Hypothesis of normal data is REJECTED by Pearson criteria at the "
                      << double(100*significance) << "% significance level.\n";
        } else {
            isNormal = true;
            std::cout << "Hypothesis of normal data is NOT REJECTED by Pearson criteria.\n";
        }
    }
    else if (data.size() > 4)
    //Kolmogorov-Smirnov Criteria
    {
        ldouble d = kolmogorov_smirnov_test(data);
        ldouble sn = std::sqrt(static_cast<ldouble>(data.size()));
        ldouble dm = d * (sn - 0.01L + 0.85L / sn);
        ldouble c = stephens_critical_value(significance);

        std::cout << "Checking normality hypothesis using Kolmogorov-Smirnov Test:\n"
        << "Size of sample (N): " << data.size() << "\n"
        << "Biggest deviation (D): " << double(d) << " \n"
        << "Transformed deviation (D*): " << double(dm) << " \n"
        << "Acceptable deviation border (C): " << double(c) << "\n";

        isNormal = (dm <= c);

        if (!isNormal) {
            std::cout << "Hypothesis of normal data is REJECTED by Kolmogorov-Smirnov criteria at the "
                      << double(100*significance) << "% significance level.\n";
        } else {
            std::cout << "Hypothesis of normal data is NOT REJECTED by Kolmogorov-Smirnov criteria.\n";
        }
    } else {
        // TODO: find other way to check normality for low amount of sample elements
        std::cout << "No implemented normality check for sample size of " << data.size() << " elements.\n";
    }

    return isNormal;
}

std::vector<ldouble> censor_normalized_data(std::vector<ldouble> data, ldouble significance) {
    std::sort(data.begin(), data.end());

    ldouble avg = average(data);
    ldouble rsmd = std::sqrt(unbiased_point_estimate_of_variance(avg, data));

    // Formula (1.2): half-width of the interval by Grubbs
    ldouble delta = grubbs_delta(data.size(), rsmd, significance);

    // Formula (1.1): the interval itself
    ldouble min = avg - delta, max = avg + delta;

    std::cout << "Censoring by Grubbs (Gaussian case):\n"
    << "Average: " << double(avg) << "\n"
    << "Deviation: " << double(rsmd) << "\n"
    << "Half-width (delta): " << double(delta) << "\n"
    << "Censoring interval: [" << double(min) << "; " << double(max) << "]\n";

    auto lo = std::lower_bound(data.begin(), data.end(), min);
    auto hi = std::upper_bound(data.begin(), data.end(), max);

    return std::vector<ldouble>(lo, hi);
}

std::vector<ldouble> censor_non_gaussian_data(std::vector<ldouble> data, ldouble significance) {
    std::sort(data.begin(), data.end());

    ldouble avgX = average(data);
    ldouble rsmdX = std::sqrt(unbiased_point_estimate_of_variance(avgX, data));
    if (rsmdX == 0) {
        std::cout << "All values are identical; there is nothing to censor.\n";
        return data;
    }

    // Formula (1.3): the simple interval that ignores the asymmetry of the data.
    // It is always reported so that both approaches can be compared.
    ldouble u4 = biased_central_statistical_moment_estimate(avgX, 4, data);
    ldouble excess = u4/std::pow(rsmdX, 4);
    ldouble deltaX = non_gaussian_delta(data.size(), rsmdX, excess);
    ldouble lowX = avgX - deltaX, highX = avgX + deltaX;

    std::cout << "Censoring of non-Gaussian data:\n"
    << "Excess: " << double(excess) << "\n"
    << "Formula (1.3) half-width (delta): " << double(deltaX) << "\n"
    << "Formula (1.3) interval: [" << double(lowX) << "; " << double(highX) << "]\n";

    johnson_parameters jp = fit_johnson(data);

    // Formula (1.4): move the sample into the normalized space
    std::vector<ldouble> normalized {};
    normalized.reserve(data.size());
    std::size_t skipped = 0;
    for (auto v : data) {
        ldouble z = johnson_transform(v, jp);
        if (std::isfinite(z))
            normalized.push_back(z);
        else
            ++skipped;
    }

    if (!jp.valid || normalized.size() < 3) {
        std::cout << "Johnson transformation could not be fitted; formula (1.3) is used.\n";
        auto lo = std::lower_bound(data.begin(), data.end(), lowX);
        auto hi = std::upper_bound(data.begin(), data.end(), highX);
        return std::vector<ldouble>(lo, hi);
    }

    ldouble avgZ = average(normalized);
    ldouble rsmdZ = std::sqrt(unbiased_point_estimate_of_variance(avgZ, normalized));

    // The methodology centers the interval at z_xbar = psi(xbar), the transformed
    // mean of the ORIGINAL data, not at the mean of the transformed values.
    ldouble centerZ = johnson_transform(avgX, jp);
    if (!std::isfinite(centerZ)) centerZ = avgZ;

    // Formulas (1.2) and (1.5): the interval is symmetric in the normalized space
    ldouble deltaZ = grubbs_delta(normalized.size(), rsmdZ, significance);
    ldouble lowZ = centerZ - deltaZ, highZ = centerZ + deltaZ;

    // Formula (1.6): back to the original scale, where it is no longer symmetric
    ldouble low  = johnson_inverse_transform(lowZ,  jp);
    ldouble high = johnson_inverse_transform(highZ, jp);

    std::cout << "Johnson normalization (formulas 1.4-1.6):\n"
    << "Family: " << johnson_family_name(jp.family) << "\n"
    << "Parameters: gamma=" << double(jp.gamma) << ", eta=" << double(jp.eta)
    << ", epsilon=" << double(jp.epsilon) << ", lambda=" << double(jp.lambda) << "\n"
    << "Normalized center (z of the mean): " << double(centerZ) << "\n"
    << "Normalized average: " << double(avgZ) << "\n"
    << "Normalized deviation: " << double(rsmdZ) << "\n"
    << "Normalized interval: [" << double(lowZ) << "; " << double(highZ) << "]\n"
    << "Johnson interval: [" << double(low) << "; " << double(high) << "]\n";

    if (skipped > 0) {
        std::cout << "  Note: " << skipped
                  << " value(s) fall outside the support of the fitted transformation "
                     "and are treated as outliers.\n";
    }

    if (!std::isfinite(low))  low  = data.front();
    if (!std::isfinite(high)) high = data.back();

    // The percentile fit is itself computed from the sample, so a single very
    // distant value can stretch the fitted tail and hide itself. Warn when the
    // normalized interval turns out much wider than the one from formula (1.3).
    if (high - low > 2*(highX - lowX)) {
        std::cout << "  Warning: the Johnson interval is more than twice as wide as the one\n"
                     "  from formula (1.3); the fit may have been distorted by extreme values.\n";
    }

    std::cout << "Censoring interval (Johnson): [" << double(low) << "; " << double(high) << "]\n";

    auto lo = std::lower_bound(data.begin(), data.end(), low);
    auto hi = std::upper_bound(data.begin(), data.end(), high);

    return std::vector<ldouble>(lo, hi);
}

std::vector<ldouble> censor_data(std::vector<ldouble> data, bool isNormal, ldouble significance) {
    if (data.size() < 3) return data;
    return isNormal
         ? censor_normalized_data(std::move(data), significance)
         : censor_non_gaussian_data(std::move(data), significance);
}
