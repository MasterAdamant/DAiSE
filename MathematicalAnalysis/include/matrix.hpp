#pragma once

#include <cmath>
#include <iostream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template<typename T>
class Matrix {
    public:
        Matrix() = default;
        Matrix(size_t rows_count, size_t columns_count) {allocate(rows_count, columns_count);}
        Matrix(std::initializer_list<std::initializer_list<T>> matrix) {
            size_t x = matrix.size();
            size_t y = (x) ? matrix.begin()->size() : 0;
            allocate(x, y);
            for (size_t i = 0; i<x; ++i) {
                for (size_t j = 0; j<y; ++j) {
                    this->matrix[i][j] = *((matrix.begin()+i)->begin()+j);
                }
            }
        }
        Matrix(std::vector<std::vector<T>> matrix) {
            size_t x = matrix.size();
            size_t y = (x) ? matrix.begin()->size() : 0;
            allocate(x, y);
            for (size_t i = 0; i<x; ++i) {
                for (size_t j = 0; j<y; ++j) {
                    this->matrix[i][j] = matrix[i][j];
                }
            }
        }
        Matrix(const Matrix<T>& other) {
            allocate(other.r_count, other.c_count);
            copy_from(other.matrix, other.r_count, other.c_count);
        }

        Matrix(Matrix<T>&& other) noexcept {swap(other);}

        Matrix<T>& operator=(Matrix<T> other) noexcept {
            swap(other);
            return *this;
        }

        ~Matrix() {release();}

        void swap(Matrix<T>& other) noexcept {
            std::swap(matrix, other.matrix);
            std::swap(r_count, other.r_count);
            std::swap(c_count, other.c_count);
            std::swap(r_cap, other.r_cap);
            std::swap(c_cap, other.c_cap);
        }

        void resize(size_t rows_count, size_t columns_count) {
            if (rows_count>r_cap || columns_count>c_cap) {
                Matrix<T> bigger {rows_count, columns_count};
                bigger.copy_from(matrix, std::min(r_count, rows_count), std::min(c_count, columns_count));
                swap(bigger);
                return;
            }
            //Clear cells that become visible again after shrinking earlier
            for (size_t i = 0; i<rows_count; ++i)
                for (size_t j = (i<r_count ? c_count : 0); j<columns_count; ++j)
                    matrix[i][j] = T{};
            r_count = rows_count;
            c_count = columns_count;
        }

        //Get element of matrix (throws std::out_of_range)
        T& operator[](size_t r, size_t c) {
            check_bounds(r, c);
            return matrix[r][c];
        }
        const T& operator[](size_t r, size_t c) const {
            check_bounds(r, c);
            return matrix[r][c];
        }

        //Get element of matrix, returns standard value (e.g. 0 for numbers) when out of range
        T get(size_t r, size_t c) const {
            if (r>=r_count || c>=c_count)
                return T{};
            return matrix[r][c];
        }

        std::vector<T> get_row(size_t index) const {
            if (index>=r_count)
                return std::vector<T>(c_count);
            return std::vector<T>(matrix[index], matrix[index]+c_count);
        }

        std::vector<T> get_column(size_t index) const {
            if (index>=c_count)
                return std::vector<T>(r_count);
            std::vector<T> result {};
            result.reserve(r_count);
            for (size_t i = 0; i<r_count; ++i)
                result.push_back(matrix[i][index]);
            return result;
        }

        std::vector<T> get_enumerated() const {
            std::vector<T> result {};
            for (size_t i = 0; i < r_count; ++i) {
                for (size_t j = 0; j < c_count; ++j) {
                    result.push_back(get(i,j));
                }
            }
            return result;
        }

        //Identity matrix of size n x n
        static Matrix<T> identity(size_t n) {
            Matrix<T> result (n, n);
            for (size_t i = 0; i<n; ++i)
                result.matrix[i][i] = T(1);
            return result;
        }

        //Determinant of a square matrix, 0x0 gives 1
        //Non-square matrices are padded with standard values to a square one (warning
        //is printed to std::cerr), so the result is 0 for them
        //Floating point types: Gaussian elimination with partial pivoting.
        //Integral types: exact fraction-free (Bareiss) elimination in std::intmax_t.
        //Other types: Bareiss elimination in T itself (needs +, -, *, exact /, == T{}).
        T determinant() const {
            if (r_count != c_count) {
                warn("determinant of a non-square matrix, padding it to a square one");
                return padded_to_square().determinant();
            }
            if constexpr (std::is_floating_point_v<T>) {
                return determinant_gauss(rows_as_vectors<T>());
            } else if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(determinant_bareiss(rows_as_vectors<std::intmax_t>()));
            } else {
                return determinant_bareiss(rows_as_vectors<T>());
            }
        }

        //Inverse of a square matrix by Gauss-Jordan elimination
        //Throws std::invalid_argument if not square (padding would always make it
        //singular), std::domain_error if singular or if T is an integral type
        //(inverse is not representable)
        Matrix<T> inverse() const {
            if (r_count != c_count)
                throw std::invalid_argument("Inverse requires a square matrix");
            if constexpr (std::is_integral_v<T>) {
                throw std::domain_error("Inverse is not representable for integral matrices");
            } else {
                const size_t n = r_count;
                Matrix<T> a {*this};
                Matrix<T> inv = identity(n);
                for (size_t k = 0; k<n; ++k) {
                    size_t pivot = n;
                    if constexpr (std::is_floating_point_v<T>) {
                        T best {};
                        for (size_t i = k; i<n; ++i) {
                            T magnitude = std::abs(a.matrix[i][k]);
                            if (magnitude>best) {best = magnitude; pivot = i;}
                        }
                    } else {
                        for (size_t i = k; i<n && pivot==n; ++i)
                            if (!(a.matrix[i][k]==T{})) pivot = i;
                    }
                    if (pivot==n)
                        throw std::domain_error("Matrix is singular");
                    std::swap(a.matrix[k], a.matrix[pivot]);
                    std::swap(inv.matrix[k], inv.matrix[pivot]);

                    T divider = a.matrix[k][k];
                    for (size_t j = 0; j<n; ++j) {
                        a.matrix[k][j] /= divider;
                        inv.matrix[k][j] /= divider;
                    }
                    for (size_t i = 0; i<n; ++i) {
                        if (i==k) continue;
                        T factor = a.matrix[i][k];
                        for (size_t j = 0; j<n; ++j) {
                            a.matrix[i][j] -= factor*a.matrix[k][j];
                            inv.matrix[i][j] -= factor*inv.matrix[k][j];
                        }
                    }
                }
                return inv;
            }
        }

        //Integer power of a square matrix by repeated squaring
        //pow(0) is the identity matrix, negative exponents use inverse()
        //Non-square matrices are padded with standard values to a square one (warning
        //is printed to std::cerr), same as in operators
        //Throws std::domain_error if a negative power is requested for a singular
        //(or integral) matrix
        Matrix<T> pow(long long exponent) const {
            if (r_count != c_count) {
                warn("power of a non-square matrix, padding it to a square one");
                return padded_to_square().pow(exponent);
            }
            Matrix<T> base {exponent<0 ? inverse() : *this};
            //Unsigned negation is well defined even for LLONG_MIN
            unsigned long long left = exponent<0 ? 0ULL-static_cast<unsigned long long>(exponent)
                                                 : static_cast<unsigned long long>(exponent);
            Matrix<T> result = identity(r_count);
            while (left) {
                if (left&1) result = result*base;
                left >>= 1;
                if (left) base = base*base;
            }
            return result;
        }

        //Math operators
        //Matrices of different sizes are summed as if padded with standard values
        Matrix<T> operator+(const Matrix<T>& value) const {
            Matrix<T> result {std::max(r_count, value.r_count), std::max(c_count, value.c_count)};
            for (size_t i = 0; i<result.r_count; ++i)
                for (size_t j = 0; j<result.c_count; ++j)
                    result.matrix[i][j] = get(i, j)+value.get(i, j);
            return result;
        }
        Matrix<T> operator-() const {
            Matrix<T> result {*this};
            for (size_t i = 0; i<result.r_count; ++i)
                for (size_t j = 0; j<result.c_count; ++j)
                    result.matrix[i][j] = -result.matrix[i][j];
            return result;
        }
        Matrix<T> operator-(const Matrix<T>& value) const {return *this+(-value);}
        Matrix<T> operator*(const Matrix<T>& value) const {
            Matrix<T> result {r_count, value.c_count};
            for (size_t i = 0; i<result.r_count; ++i) {
                for (size_t j = 0; j<result.c_count; ++j) {
                    for (size_t k = 0; k<c_count; ++k) {
                        result[i,j] += get(i,k)*value.get(k,j);
                    }
                }
            }
            return result;
        }
        template<typename E>
        Matrix<T> operator*(const E& coefficient) const {
            Matrix<T> result {*this};
            for (size_t i=0; i<r_count; ++i) {
                for (size_t j=0; j<c_count; ++j) {
                    result[i,j] *= coefficient;
                }
            }
            return result;
        }
        template<typename E>
        Matrix<T> operator/(const E& coefficient) const {
            Matrix<T> result {*this};
            for (size_t i=0; i<r_count; ++i) {
                for (size_t j=0; j<c_count; ++j) {
                    result[i,j] /= coefficient;
                }
            }
            return result;
        }

        //Self-applied operators
        Matrix<T> operator+=(const Matrix<T>& value) {
            return (*this = *this + value);
        }
        Matrix<T> operator-=(const Matrix<T>& value) {
            return (*this = *this - value);
        }
        Matrix<T> operator*=(const Matrix<T>& value) {
            return (*this = *this * value);
        }
        template<typename E>
        Matrix<T>& operator*=(const E& coefficient) {
            return (*this = *this * coefficient);
        }
        template<typename E>
        Matrix<T>& operator/=(const E& coefficient) {
            return (*this = *this / coefficient);
        }

        //Transposed matrix
        Matrix<T> operator*() const {
            Matrix<T> result {c_count, r_count};
            for (size_t i = 0; i<r_count; ++i)
                for (size_t j = 0; j<c_count; ++j)
                    result.matrix[j][i] = matrix[i][j];
            return result;
        }

        //Logging
        friend std::ostream& operator<<(std::ostream& stream, const Matrix<T>& value) {
            if (!value.r_count || !value.c_count) {
                stream << "{ }";
                return stream;
            }

            stream << "{ ";
            for (size_t i = 0; i<value.r_count-1; ++i) {
                stream << '{';
                for (size_t j = 0; j<value.c_count-1; ++j) {
                    stream << value[i,j] << ", ";
                }
                stream << value[i,value.c_count-1] << "}, ";
            }

            stream << '{';
            for (size_t j = 0; j<value.c_count-1; ++j) {
                stream << value[value.r_count-1,j] << ", ";
            }
            stream << value[value.r_count-1,value.c_count-1] << "} }";
            
            return stream;
        }
        

        //Sizes and capacity
        size_t rows_count() const {return r_count;}
        size_t columns_count() const {return c_count;}
        size_t rows_capacity() const {return r_cap;}
        size_t columns_capacity() const {return c_cap;}

    private:
        T** matrix = nullptr;
        size_t r_count = 0, c_count = 0;
        size_t r_cap = 0, c_cap = 0;
        static constexpr size_t CAPACITY_COEFFICIENT = 4;

        //Smallest multiple of denominator strictly greater than size
        static size_t get_capacity_by_size(size_t size, size_t denominator) {
            return (size/denominator+1)*denominator;
        }

        void check_bounds(size_t r, size_t c) const {
            if (r>=r_count || c>=c_count)
                throw std::out_of_range("Matrix index out of range");
        }

        //Allocates zero-initialised storage; expects matrix to be empty
        void allocate(size_t x, size_t y) {
            size_t rows_cap = get_capacity_by_size(x, CAPACITY_COEFFICIENT);
            size_t cols_cap = get_capacity_by_size(y, CAPACITY_COEFFICIENT);

            T** memory = new T*[rows_cap]{};
            try {
                for (size_t i = 0; i<rows_cap; ++i)
                    memory[i] = new T[cols_cap]{};
            } catch (...) {
                for (size_t i = 0; i<rows_cap; ++i)
                    delete[] memory[i];
                delete[] memory;
                throw;
            }

            matrix = memory;
            r_count = x;
            c_count = y;
            r_cap = rows_cap;
            c_cap = cols_cap;
        }

        void release() noexcept {
            if (matrix)
                for (size_t i = 0; i<r_cap; ++i)
                    delete[] matrix[i];
            delete[] matrix;
            matrix = nullptr;
            r_count = c_count = r_cap = c_cap = 0;
        }

        static void warn(const char* message) {
            std::cerr << "Matrix warning: " << message << '\n';
        }

        //Copy extended with standard values to max(rows, columns) x max(rows, columns)
        Matrix<T> padded_to_square() const {
            const size_t n = std::max(r_count, c_count);
            Matrix<T> result (n, n);
            result.copy_from(matrix, r_count, c_count);
            return result;
        }

        template<typename W>
        std::vector<std::vector<W>> rows_as_vectors() const {
            std::vector<std::vector<W>> result (r_count);
            for (size_t i = 0; i<r_count; ++i)
                result[i].assign(matrix[i], matrix[i]+c_count);
            return result;
        }

        template<typename W>
        static W determinant_gauss(std::vector<std::vector<W>> a) {
            const size_t n = a.size();
            W result(1);
            for (size_t k = 0; k<n; ++k) {
                size_t pivot = k;
                for (size_t i = k+1; i<n; ++i)
                    if (std::abs(a[i][k])>std::abs(a[pivot][k])) pivot = i;
                if (a[pivot][k]==W{})
                    return W{};
                if (pivot!=k) {
                    std::swap(a[pivot], a[k]);
                    result = -result;
                }
                result *= a[k][k];
                for (size_t i = k+1; i<n; ++i) {
                    W factor = a[i][k]/a[k][k];
                    for (size_t j = k+1; j<n; ++j)
                        a[i][j] -= factor*a[k][j];
                }
            }
            return result;
        }

        template<typename W>
        static W determinant_bareiss(std::vector<std::vector<W>> a) {
            const size_t n = a.size();
            if (n==0) return W(1);
            bool negate = false;
            W previous(1);
            for (size_t k = 0; k+1<n; ++k) {
                if (a[k][k]==W{}) {
                    size_t pivot = k+1;
                    while (pivot<n && a[pivot][k]==W{}) ++pivot;
                    if (pivot==n) return W{};
                    std::swap(a[pivot], a[k]);
                    negate = !negate;
                }
                for (size_t i = k+1; i<n; ++i)
                    for (size_t j = k+1; j<n; ++j)
                        a[i][j] = (a[i][j]*a[k][k]-a[i][k]*a[k][j])/previous;
                previous = a[k][k];
            }
            return negate ? -a[n-1][n-1] : a[n-1][n-1];
        }

        void copy_from(T const* const* values, size_t x, size_t y) {
            if (!matrix || !values) return;
            for (size_t i = 0; i<std::min(r_count, x); ++i)
                for (size_t j = 0; j<std::min(c_count, y); ++j)
                    matrix[i][j] = values[i][j];
        }
};