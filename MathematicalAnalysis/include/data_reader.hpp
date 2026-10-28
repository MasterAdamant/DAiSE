#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.hpp"

class DataReader {
    public:
    DataReader() = delete; //Making class static

    static constexpr const char* whitespace = " \t\r\n";

    template<typename T>
    static std::vector<T> StrToSet(const std::string& source, const std::string& delimiters = whitespace) {
        std::vector<T> result;

        std::size_t pos = source.find_first_not_of(delimiters);
        while (pos != std::string::npos) {
            std::size_t border = source.find_first_of(delimiters, pos);
            if (border == std::string::npos) border = source.size();

            std::istringstream val {source.substr(pos, border - pos)};
            T element;
            if (val >> element) {
                result.push_back(element);
            }

            pos = source.find_first_not_of(delimiters, border);
        }

        return result;
    }

    static std::string ReadRaw(const std::string& filepath) {
        std::ifstream file {filepath};
        if (!file.is_open())
            return {};

        std::ostringstream buffer;
        std::string line;
        while (std::getline(file, line)) {
            buffer << line << ' ';
        }
        return buffer.str();
    }

    static std::string NormalizeDecimalComma(std::string source) {
        std::replace(source.begin(), source.end(), ',', '.');
        return source;
    }

    template<typename T>
    static std::vector<T> ReadVector(const std::string& filepath, const std::string& delimiters = whitespace) {
        return StrToSet<T>(NormalizeDecimalComma(ReadRaw(filepath)), delimiters);
    }

    template<typename T>
    static Matrix<T> ReadMatrix(const std::string& filepath, const std::string& delimiters = whitespace) {
        std::ifstream file {filepath};
        if (!file.is_open())
            return Matrix<T>{};

        std::vector<std::vector<T>> rows;
        std::string line;
        std::size_t width = 0;
        while (std::getline(file, line)) {
            std::vector<T> row = StrToSet<T>(NormalizeDecimalComma(line), delimiters);
            if (row.empty()) continue;
            width = std::max(width, row.size());
            rows.push_back(std::move(row));
        }

        if (rows.empty())
            return Matrix<T>{};

        Matrix<T> result {rows.size(), width};
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (std::size_t j = 0; j < rows[i].size(); ++j)
                result[i, j] = rows[i][j];

        return result;
    }

    template<typename T>
    static std::vector<std::vector<T>> ReadColumns(const std::string& filepath, const std::string& delimiters = whitespace) {
        Matrix<T> matrix = ReadMatrix<T>(filepath, delimiters);

        std::vector<std::vector<T>> columns;
        columns.reserve(matrix.columns_count());
        for (std::size_t j = 0; j < matrix.columns_count(); ++j) {
            columns.push_back(matrix.get_column(j));
        }
        return columns;
    }
};
