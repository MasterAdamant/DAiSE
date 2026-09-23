# Lab 1 — Statistical Analysis and Censoring of a Sample

A command-line tool that reads a numeric sample, computes its descriptive
statistics, plots a histogram with a fitted normal density, tests the
hypothesis that the sample is normally distributed, and then iteratively
censors (removes) outliers until the sample stabilizes.

## Build

Requires CMake ≥ 3.16 and a C++20 compiler.

```sh
cmake -B build -S .
cmake --build build
```

This produces the `Lab1` executable in the project root.

## Usage

```sh
./Lab1 <path-to-data-file>
```

The input file may contain any number of real values separated by spaces
and/or newlines (see `data.txt` for an example). At least 3 values are
required.

Example:

```sh
./Lab1 data.txt
```

## What the program does

1. **Read & sort.** `ReadData` parses whitespace-separated numbers from the
   file into a `vector<long double>`; the sample is then sorted ascending.
2. **Precalculation.** For the current sample it computes the average,
   the unbiased variance estimate, the standard deviation, and the number of
   histogram sections (Sturges' formula).
3. **Histogram.** `write_histogram_svg` renders the sample's histogram with
   the fitted normal density curve overlaid, as a standalone SVG file named
   `histogram_pass<N>.svg` (`N` = 0 for the original sample, 1, 2, … for each
   censoring pass).
4. **Normality check** (`gaussian_normality_check`):
   - For samples with **more than 30** values — Pearson's chi-squared
     goodness-of-fit test.
   - For samples with **5–30** values — the Kolmogorov–Smirnov test with
     Stephens' correction.
   - For **4 or fewer** values, no test is performed.
   If the sample is found normal, the trusted range for the mean and for the
   deviation is also printed.
5. **Censoring** (`censor_data`), repeated until a pass removes nothing or
   fewer than 3 values remain:
   - If the sample is normal, `censor_normalized_data` discards values
     outside `[average ± Grubbs delta]` (formulas 1.1–1.2).
   - If it is not normal, `censor_non_gaussian_data` fits a Johnson
     normalizing transformation, censors the sample in the transformed
     (normal) space with the same Grubbs approach, and maps the interval
     back to the original scale (formulas 1.4–1.6). If the transformation
     cannot be fitted, it falls back to a simpler interval that ignores
     skewness (formula 1.3).
   - After each successful censoring pass the normality check and histogram
     are repeated on the reduced sample.
6. **Conclusions.** Once the loop stops, the program prints how many values
   were kept vs. discarded and whether the final sample can be considered
   normally distributed at the chosen significance level.

The significance level used throughout is `0.05` (5%), set in `main.cpp`.

## Project layout

```
include/
  analysis.hpp   Histogram rendering, normality check, censoring — declarations
  formulas.hpp   Statistical formulas and distribution helpers — declarations
src/
  main.cpp       Entry point: read data, drive the analyze/censor loop, print results
  analysis.cpp   Histogram SVG rendering, Pearson/Kolmogorov-Smirnov checks, censoring
  formulas.cpp   Descriptive statistics, distribution CDFs/quantiles, Johnson fit
data.txt         Example input sample
```

## Output

Console output includes, for the original sample and after every censoring
pass: precalculated statistics, the normality test details and verdict, and
(when applicable) the trusted ranges for the mean and deviation. Each pass
also writes its own histogram SVG (`histogram_pass0.svg`, `histogram_pass1.svg`, …)
that can be opened directly in a browser.
