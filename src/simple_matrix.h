/*
    MIT License

    Copyright (c) 2021 ETCHKILI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/**
 * @file simple_matrix.cpp
 * @author Guo Yubin, Shen Zhilong, Fan Leilong
 * @date 2021-06-11
 * @Github https://github.com/ETCHKILI/cs205-project-simple-matrix
 * @Organization SUSTech
 * @version 0.1.0
 * @attention Should specify the OpenCV(compiled by mingw) path and Eigen path
 * in the CMakeLists.txt. For detail please see CmakeLists.txt
 */
#ifndef CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H
#define CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H

#include "simple_matrix_exception.h"

#include <cstdint>
#include <iterator>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include <typeinfo>

/**
 * @brief The namespace of the simple_matrix project, including one
 * matrix class and three exception class
 */
namespace simple_matrix {

    static const constexpr int64_t kDefaultRowSize = 10;
    static const constexpr int64_t kDefaultColumnSize = 10;
    static const constexpr int64_t kDefaultSideLength = 10;
    static const uint64_t kMaxAllocateSize = UINT64_MAX;
    static const int64_t kOSBits = sizeof (void *) * 8;
    static const double eps = 1e-3;

    /**
     * @brief Enum class that specify the direction of the vector(s):
     * Row vector, Column vector or a Sub-matrix
     */
    enum class SelectAs {
        SUB_MATRIX, ROW, COLUMN
    };


    /**
     * @class Matrix
     * @brief Base class in simple_matrix
     * @tparam T
     * @details Use one-dimension array to simulate a matrix
     * store the data in continuous heap memory which make it efficient
     */
    template<typename T>
    class Matrix {
    private:

    protected:
        /**
         * @brief the pointer to the data array
         */
        T *data_;

        uint64_t row_size_; ///< the row size
        uint64_t column_size_; ///< the column size

        explicit Matrix();

    public:
        T *operator[](int row) const;

        explicit Matrix(uint64_t row_size, uint64_t column_size);

        explicit Matrix(uint64_t row_size, uint64_t column_size, const T &initial_value);

        explicit Matrix(const std::vector<T> &v, SelectAs selectAs);

        explicit operator cv::Mat_<T>();

        Matrix(const Matrix<T> &that);

        Matrix(Matrix<T> &&that) noexcept;

        Matrix<T> &operator=(const Matrix<T> &that);

        Matrix<T> &operator=(Matrix<T> &&that) noexcept;

        virtual ~Matrix();

        virtual T &Access(uint64_t row, uint64_t column);

        virtual void SetValue(T val);

        [[nodiscard]] uint64_t getRowSize() const;

        [[nodiscard]] uint64_t getColumnSize() const;

        static bool CheckSizeValid(uint64_t row_size, uint64_t column_size);

        [[nodiscard]] Matrix<T> operator+(const Matrix<T> &that);

        [[nodiscard]] Matrix<T> operator+(const T &k);

        Matrix<T> &operator+=(const Matrix<T> &that);

        Matrix<T> &operator+=(const T &k);

        [[nodiscard]] Matrix<T> operator-(const Matrix<T> &that);

        [[nodiscard]] Matrix<T> operator-(const T &k);

        Matrix<T> &operator-=(const Matrix<T> &that);

        Matrix<T> &operator-=(const T &k);

        [[nodiscard]] Matrix<T> operator*(const Matrix<T> &that);

        [[nodiscard]] Matrix<T> operator*(const T &k);

        Matrix<T> &operator*=(const Matrix<T> &that);

        Matrix<T> &operator*=(const T &k);

        [[nodiscard]] Matrix<T> operator/(const Matrix<T> &that);

        [[nodiscard]] Matrix<T> operator/(const T &k);

        Matrix<T> &operator/=(const Matrix<T> &that);

        Matrix<T> &operator/=(const T &k);

        T FindMin(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX,
                  std::function<bool(const T &, const T &)> compare = [](const T &a, const T &b) { return a < b; });

        T FindMax(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX,
                  std::function<bool(const T &, const T &)> compare = [](const T &a, const T &b) { return a > b; });

        T Sum(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX);

        T Average(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX);

        Matrix<T> getRotate180();

        Matrix<T> subMatrix(uint64_t row_lo, uint64_t col_lo, uint64_t row_hi, uint64_t col_hi);

        Matrix<T> convolution(Matrix<T> &that);

        [[nodiscard]] Matrix<T> transpose();

        static Matrix<T> identity(uint64_t s, T t);

        [[nodiscard]] Matrix<double> Householder(uint64_t col, uint64_t ele);

        [[nodiscard]] Matrix<double> Hessenberg();

        [[nodiscard]] Matrix<double> Givens(uint64_t col, uint64_t begin, uint64_t end);

        [[nodiscard]] Matrix<double> QR_iteration();

        [[nodiscard]] std::vector<double> eigenvalue();

        [[nodiscard]] Matrix<T> inverse();

        std::vector<std::pair<T, std::vector<T>>> eigen();

        [[nodiscard]] T trace();

        [[nodiscard]] T determinant();

        [[nodiscard]] Matrix<T> reshape(int32_t row, int32_t col);

        [[nodiscard]] Matrix<T> slice(int32_t row1, int32_t row2, int32_t col1, int32_t col2);

        [[nodiscard]] Matrix<T> Conjugate(std::function<T(const T &)> conjugate);
    };

    /**
     * @protected because it is not meaningful to create a 1-row-1-column
     * matrix. If it is needed, you can specify the size.
     * @tparam T
     */
    template<typename T>
    Matrix<T>::Matrix() {
        Matrix(1, 1);
    }

    /**
     * @brief Constructor that initialize the element with default constructor
     * @tparam T
     * @param row_size
     * @param column_size
     */
    template<typename T>
    Matrix<T>::Matrix(uint64_t row_size, uint64_t column_size) {
        if (!CheckSizeValid(row_size, column_size)) {
            throw simple_matrix::BadSizeException("Size invalid!");
        }
        data_ = new T[row_size * column_size]();
        column_size_ = column_size;
        row_size_ = row_size;
    }

    /*!
     * @brief Constructor that initialize the data
     * @tparam T
     * @param row_size
     * @param column_size
     * @param initial_value
     */
    template<typename T>
    Matrix<T>::Matrix(uint64_t row_size, uint64_t column_size, const T &initial_value) {
        if (!CheckSizeValid(row_size, column_size)) {
            throw simple_matrix::BadSizeException("Size too large!");
        }
        data_ = new T[row_size * column_size]();
        column_size_ = column_size;
        row_size_ = row_size;
        for (int i = 0; i < row_size_; ++i) {
            for (int j = 0; j < column_size_; ++j) {
                data_[i * column_size_ + j] = initial_value;
            }
        }
    }

    /**
     * @brief Constructor by a vector
     * @tparam T
     * @param v
     * @param selectAs
     */
    template<typename T>
    Matrix<T>::Matrix(const std::vector<T> &v, SelectAs selectAs) {
        if (v.empty()) {
            throw BadSizeException("Size zero error");
        }
        data_ = new T[v.size()];
        if (selectAs == SelectAs::COLUMN) {
            column_size_ = v.size();
            row_size_ = 1;
        } else {
            column_size_ = 1;
            row_size_ = v.size();
        }
    }

    /**
     * @brief Copy constructor
     * @tparam T
     * @param that
     */
    template<typename T>
    Matrix<T>::Matrix(const Matrix<T> &that) {
        data_ = new T[that.row_size_ * that.column_size_];
        row_size_ = that.row_size_;
        column_size_ = that.column_size_;
        int n = row_size_ * column_size_;
        for (int i = 0; i < n; ++i) {
            data_[i] = that.data_[i];
        }
    }

    /**
     * @brief Move constructor
     * @tparam T
     * @param that
     */
    template<typename T>
    Matrix<T>::Matrix(Matrix<T> &&that) noexcept {
        data_ = that.data_;
        row_size_ = that.row_size_;
        column_size_ = that.column_size_;
        that.data_ = nullptr;
    }

    /**
    * @brief Destructor : delete []data
    * @tparam T
    * @param row_size
    * @param column_size
    * @param initial_value
    */
    template<typename T>
    Matrix<T>::~Matrix() {
        if (data_ != nullptr) {
            delete[]data_;
        }
    }

    /**
     * @brief Copy assignment
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator=(const Matrix<T> &that) {
        if (this != &that) {
            data_ = new T[that.row_size_ * that.column_size_];
            row_size_ = that.row_size_;
            column_size_ = that.column_size_;
            int n = row_size_ * column_size_;
            for (int i = 0; i < n; ++i) {
                data_[i] = that.data_[i];
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator=(Matrix<T> &&that) noexcept {
        if (this != &that) {
            data_ = that.data_;
            row_size_ = that.row_size_;
            column_size_ = that.column_size_;
            that.data_ = nullptr;
        }
        return *this;
    }


    /**
     * @brief The operator[] which can access the data
     * @detail return a pointer equal to (data_ + row * column_size_)
     * @example use like this: {@code matrix[i][j]} to access the data in row i and column j
     * @tparam T
     * @param row
     * @return
     */
    template<typename T>
    T *Matrix<T>::operator[](int row) const {
        return data_ + row * column_size_;
    }

    /**
     * @brief Check if the size is valid for constructing a matrix
     * @tparam T
     * @param row_size
     * @param column_size
     * @return True if the size is valid for constructing a matrix. False if not
     */
    template<typename T>
    bool Matrix<T>::CheckSizeValid(uint64_t row_size, uint64_t column_size) {
        return ((row_size <= kMaxAllocateSize / column_size) || (column_size <= kMaxAllocateSize / row_size)) &&
               std::max(row_size, column_size) > 0;
    }

    /**
     * @brief Set the value for the whole matrix
     * @tparam T
     * @param val
     */
    template<typename T>
    void Matrix<T>::SetValue(T val) {
        if (data_ == nullptr) {
            throw BadAccessException("Matrix is not initialized or index out of range");
        }
        for (int i = 0; i < row_size_; ++i) {
            for (int j = 0; j < column_size_; ++j) {
                data_[i * column_size_ + j] = val;
            }
        }
    }

    /**
     * @brief The way that can be overloaded by derived class to access the data (get or change the value)
     * @tparam T
     * @param row
     * @param column
     * @return
     */
    template<typename T>
    T &Matrix<T>::Access(uint64_t row, uint64_t column) {
        if (row > row_size_ - 1 || column > column_size_ - 1) {
            throw BadAccessException("Index out of bounds");
        }
        return (*this)[row][column];
    }

    /**
     * @brief the public way to get the row_size of the matrix
     * @tparam T
     * @return
     */
    template<typename T>
    uint64_t Matrix<T>::getRowSize() const {
        return row_size_;
    }

    /**
     * @brief the public way to get the column_size of the matrix
     * @tparam T
     * @return
     */
    template<typename T>
    uint64_t Matrix<T>::getColumnSize() const {
        return column_size_;
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that: another matrix
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T> &that) {
        if (this->column_size_ != that.column_size_ || this->row_size_ != that.row_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T> result(this->row_size_, this->column_size_);
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] + that.data_[i];
        }
        return result;
    }

    /**
     * @brief operator overload, every element of the matrix should be operate with the value
     * @tparam T
     * @param k: a value
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator+(const T &k) {
        Matrix<T> result(this->row_size_, this->column_size_);
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] + k;
        }
        return result;
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &that) {
        if (this->column_size_ != that.column_size_ || this->row_size_ != that.row_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            this->data_[i] = this->data_[i] + that.data_[i];
        }
        return (*this);
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param k
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator+=(const T &k) {
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            this->data_[i] = this->data_[i] + k;
        }
        return (*this);
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that: another matrix
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T> &that) {
        if (this->column_size_ != that.column_size_ || this->row_size_ != that.row_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T> result(this->row_size_, this->column_size_);
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] - that.data_[i];
        }
        return result;
    }

    /**
     * @brief operator overload, every element of the matrix should be operate with the value
     * @tparam T
     * @param k: a value
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator-(const T &k) {
        Matrix<T> result(this->row_size_, this->column_size_);
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] - k;
        }
        return result;
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &that) {
        if (this->column_size_ != that.column_size_ || this->row_size_ != that.row_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            this->data_[i] = this->data_[i] - that.data_[i];
        }
        return (*this);
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param k
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator-=(const T &k) {
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            this->data_[i] = this->data_[i] - k;
        }
        return (*this);
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that: another matrix
     * @detail use continuous memory access to speed up the cross multiplication
     * @return the result of cross product of two matrix
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T> &that) {
        if (this->column_size_ != that.row_size_ || this->row_size_ != that.column_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        int n = this->row_size_;
        Matrix<T> result(n, n);
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < this->column_size_; ++k) {
                T temp = (*this)[i][k];
                for (int j = 0; j < n; ++j) {
                    result[i][j] = result[i][j] + temp * that[k][j];
                }
            }
        }
        return result;
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param k
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator*(const T &k) {
        Matrix<T> result(this->row_size_, this->column_size_);
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] * k;
        }
        return result;
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &that) {
        (*this) = (*this) * that;
        return (*this);
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param k
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator*=(const T &k) {
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            this->data_[i] = this->data_[i] * k;
        }
        return (*this);
    }

    /**
     * @brief Division between matrix, A / B is calculated as A * B.inverse
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator/(const Matrix<T> &that) {
        return (*this) * that.inverse();
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param k
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::operator/(const T &k) {
        Matrix<T> result(this->row_size_, this->column_size_);
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] / k;
        }
        return result;
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator/=(const Matrix<T> &that) {
        (*this) = (*this) / that;
        return (*this);
    }

    /**
     * @brief operator overload
     * @tparam T
     * @param that
     * @return
     */
    template<typename T>
    Matrix<T> &Matrix<T>::operator/=(const T &k) {
        uint64_t n = this->row_size_ * this->column_size_;
        for (int i = 0; i < n; ++i) {
            this->data_[i] = this->data_[i] / k;
        }
        return (*this);
    }

    /**
     * @brief Implement the convolution of the matrix, supported by the operator overloads
     * @tparam T
     * @param that
     * @return the result matrix
     */
    template<typename T>
    Matrix<T> Matrix<T>::convolution(Matrix<T> &that) {
        uint64_t new_row_size = this->row_size_ + that.row_size_ - 1;
        uint64_t new_col_size = this->column_size_ + that.column_size_ - 1;
        Matrix<T> res(new_row_size, new_col_size);
        Matrix<T> thatRotated = that.getRotate180();
        uint64_t ROW_lo, ROW_hi, COL_lo, COL_hi;
        for (uint64_t ROW = 0; ROW < new_row_size; ++ROW) {
            for (uint64_t COL = 0; COL < new_col_size; ++COL) {
                ROW_lo = (that.row_size_ - 1 > ROW)
                         ? that.row_size_ - 1 : ROW;
                COL_lo = (that.column_size_ - 1 > COL)
                         ? that.column_size_ - 1 : COL;
                ROW_hi = (that.row_size_ + this->row_size_ - 2 < ROW + that.row_size_ - 1)
                         ? that.row_size_ + this->row_size_ - 2 : ROW + that.row_size_ - 1;
                COL_hi = (that.column_size_ + this->column_size_ - 2 < COL + that.column_size_ - 1)
                         ? that.column_size_ + this->column_size_ - 2 : COL + that.column_size_ - 1;
                auto a = this->subMatrix(ROW_lo - that.row_size_ + 1,
                                         COL_lo - that.column_size_ + 1,
                                         ROW_hi - that.row_size_ + 1,
                                         COL_hi - that.column_size_ + 1);
                auto b = thatRotated.subMatrix(ROW_lo - ROW,
                                               COL_lo - COL,
                                               ROW_hi - ROW,
                                               COL_hi - COL);
                res[ROW][COL] = InnerProduct(a, b);
            }
        }
        return res;
    }

    /**
     * @brief Implement the innerProduct of the matrix
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto InnerProduct(Matrix<T1> a, Matrix<T2> b) {
        using T3 = decltype(std::declval<T1>() * std::declval<T2>());
        auto na = a.getRowSize();
        auto ma = a.getColumnSize();
        auto nb = b.getRowSize();
        auto mb = b.getColumnSize();

        if (na != nb || ma != mb) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        T3 innerProduct = a.Access(0, 0) * b.Access(0, 0);
        for (int i = 1; i < na * ma; ++i) {
            innerProduct += a.Access(i / ma, i % ma) * b.Access(i / mb, i % mb);
        }
        return innerProduct;
    }

    /**
     * @brief the sub-matrix specified by the index
     * @tparam T
     * @param row_lo
     * @param col_lo
     * @param row_hi
     * @param col_hi
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::subMatrix(uint64_t row_lo, uint64_t col_lo, uint64_t row_hi, uint64_t col_hi) {
        uint64_t new_row_size = row_hi - row_lo + 1;
        uint64_t new_col_size = col_hi - col_lo + 1;
        Matrix<T> res(new_row_size, new_col_size);
        for (int i = 0; i < new_row_size; ++i) {
            for (int j = 0; j < new_col_size; ++j) {
                res.Access(i, j) = Access(row_lo + i, col_lo + j);
            }
        }
        return res;
    }

    /**
     * @brief Get a matrix which is the result of 180-rotation of this matrix
     * @tparam T
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::getRotate180() {
        Matrix<T> res(row_size_, column_size_);
        for (int i = 0; i < row_size_ * column_size_; ++i) {
            res.Access(i / column_size_, i % column_size_) = Access(row_size_ - 1 - (int) (i / column_size_),
                                                                    column_size_ - 1 - i % column_size_);
        }
        return res;
    }

    /**
     * @brief Find the min element, in the matrix (or rows or columns or sub-matrix, you
     * can specify the index by a vector)
     * @tparam T
     * @param index : the vector that specify the index
     * @param selectAs
     * @param compare
     * @return
     */
    template<typename T>
    T Matrix<T>::FindMin(std::vector<int32_t> index, SelectAs selectAs,
                         std::function<bool(const T &, const T &)> compare) {
        T res;
        if (selectAs == SelectAs::COLUMN) {
            res = data_[index[0]];
            for (int i = 0; i < index.size(); ++i) {
                for (int j = 0; j < row_size_; ++j) {
                    if (compare(data_[j * column_size_ + index[i]], res)) {
                        res = data_[j * column_size_ + index[i]];
                    }
                }
            }
        } else if (selectAs == SelectAs::ROW) {
            res = data_[index[0] * column_size_];
            for (int i = 0; i < index.size(); ++i) {
                for (int j = 0; j < column_size_; ++j) {
                    if (compare(data_[index[i] * column_size_ + j], res)) {
                        res = data_[index[i] * column_size_ + j];
                    }
                }
            }
        } else {
            if (index.empty()) {
                index = std::vector<int32_t>();
                index.push_back(0);
                index.push_back(0);
                index.push_back(row_size_);
                index.push_back(column_size_);
            }
            res = data_[index[0] * column_size_ + index[1]];
            for (int i = index[0]; i < index[2]; ++i) {
                for (int j = index[1]; j < index[3]; ++j) {
                    if (compare(data_[i * column_size_ + j], res)) {
                        res = data_[i * column_size_ + j];
                    }
                }
            }
        }
        return res;
    }

    /**
     * @brief Find the max element, in the matrix (or rows or columns or sub-matrix, you
     * can specify the index by a vector)
     * @tparam T
     * @param index : the vector that specify the index
     * @param selectAs
     * @param compare
     * @return
     */
    template<typename T>
    T Matrix<T>::FindMax(std::vector<int32_t> index, SelectAs selectAs,
                         std::function<bool(const T &, const T &)> compare) {
        T res;
        if (selectAs == SelectAs::COLUMN) {
            res = data_[index[0]];
            for (int i = 0; i < index.size(); ++i) {
                for (int j = 0; j < row_size_; ++j) {
                    if (compare(data_[j * column_size_ + index[i]], res)) {
                        res = data_[j * column_size_ + index[i]];
                    }
                }
            }
        } else if (selectAs == SelectAs::ROW) {
            res = data_[index[0] * column_size_];
            for (int i = 0; i < index.size(); ++i) {
                for (int j = 0; j < column_size_; ++j) {
                    if (compare(data_[index[i] * column_size_ + j], res)) {
                        res = data_[index[i] * column_size_ + j];
                    }
                }
            }
        } else {
            if (index.empty()) {
                index = std::vector<int32_t>();
                index.push_back(0);
                index.push_back(0);
                index.push_back(row_size_);
                index.push_back(column_size_);
            }
            res = data_[index[0] * column_size_ + index[1]];
            for (int i = index[0]; i < index[2]; ++i) {
                for (int j = index[1]; j < index[3]; ++j) {
                    if (compare(data_[i * column_size_ + j], res)) {
                        res = data_[i * column_size_ + j];
                    }
                }
            }
        }
        return res;
    }

    /**
     * @brief Sum the min element, in the matrix (or rows or columns or sub-matrix, you
     * can specify the index by a vector)
     * @tparam T
     * @param index : the vector that specify the index
     * @param selectAs
     * @param compare
     * @return
     */
    template<typename T>
    T Matrix<T>::Sum(std::vector<int32_t> index, SelectAs selectAs) {
        T res;
        if (selectAs == SelectAs::COLUMN) {
            res = data_[index[0]];
            for (int i = 0; i < index.size(); ++i) {
                for (int j = 0; j < row_size_; ++j) {
                    if (i != 0 || j != 0) {
                        res = res + data_[j * column_size_ + index[i]];
                    }
                }
            }
        } else if (selectAs == SelectAs::ROW) {
            res = data_[index[0] * column_size_];
            for (int i = 0; i < index.size(); ++i) {
                for (int j = 0; j < column_size_; ++j) {
                    if (i != 0 || j != 0) {
                        res = res + data_[index[i] * column_size_ + j];
                    }
                }
            }
        } else {
            if (index.empty()) {
                index = std::vector<int32_t>();
                index.push_back(0);
                index.push_back(0);
                index.push_back(row_size_);
                index.push_back(column_size_);
            }
            res = data_[index[0] * column_size_ + index[1]];
            for (int i = index[0]; i < index[2]; ++i) {
                for (int j = index[1]; j < index[3]; ++j) {
                    if (i != index[0] || j != index[1]) {
                        res = res + data_[i * column_size_ + j];
                    }
                }
            }
        }
        return res;
    }

    /**
     * @brief Find the average, in the matrix (or rows or columns or sub-matrix, you
     * can specify the index by a vector)
     * @tparam T
     * @param index : the vector that specify the index
     * @param selectAs
     * @param compare
     * @return
     */
    template<typename T>
    T Matrix<T>::Average(std::vector<int32_t> index, SelectAs selectAs) {
        return (Sum(index, selectAs) / (row_size_ * column_size_));
    }

    /**
     * @brief operator overload, extra support added if two types can be operated.
     * For example, if T1 + T2 is valid, then Matrix<T1> + Matrix<T2> is valid
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto operator+(Matrix<T1> &a, Matrix<T2> &b) {
        using T3 = decltype(std::declval<T1>() + std::declval<T2>());
        auto na = a.getRowSize();
        auto ma = a.getColumnSize();
        auto nb = b.getRowSize();
        auto mb = b.getColumnSize();

        if (na != nb || ma != mb) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T3> c(na, ma);
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < ma; ++j) {
                c.Access(i, j) = a.Access(i, j) + b.Access(i, j);
            }
        }
        return c;
    }

    /**
     * @brief operator overload, extra support added if two types can be operated.
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto operator-(Matrix<T1> &a, Matrix<T2> &b) {
        using T3 = decltype(std::declval<T1>() - std::declval<T2>());
        auto na = a.getRowSize();
        auto ma = a.getColumnSize();
        auto nb = b.getRowSize();
        auto mb = b.getColumnSize();

        if (na != nb || ma != mb) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T3> c(na, ma);
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < ma; ++j) {
                c.Access(i, j) = a.Access(i, j) - b.Access(i, j);
            }
        }
        return c;
    }

    /**
     * @brief operator overload, extra support added if two types can be operated.
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto operator*(Matrix<T1> &a, Matrix<T2> &b) {
        using T3 = decltype(std::declval<T1>() * std::declval<T2>());
        auto na = a.getRowSize();
        auto ma = a.getColumnSize();
        auto nb = b.getRowSize();
        auto mb = b.getColumnSize();

        if (na != mb || ma != nb) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T3> c(na, na);
        for (int i = 0; i < na; ++i) {
            for (int k = 0; k < ma; ++k) {
                T1 temp = a.Access(i, k);
                for (int j = 0; j < na; ++j) {
                    c.Access(i, j) = c.Access(i, j) + temp * b.Access(k, j);
                }
            }
        }
        return c;
    }

    /**
     * @brief Implement the dot multiplication
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto DotMultiply(Matrix<T1> &a, Matrix<T2> &b) {
        using T3 = decltype(std::declval<T1>() * std::declval<T2>());
        auto na = a.getRowSize();
        auto ma = a.getColumnSize();
        auto nb = b.getRowSize();
        auto mb = b.getColumnSize();

        if (na != nb || ma != mb) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T3> c(na, ma);
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < ma; ++j) {
                c.Access(i, j) = a.Access(i, j) * b.Access(i, j);
            }
        }
        return c;
    }

    /**
     * @brief operator overload, extra support added if two types can be operated.
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto operator*(Matrix<T1> &a, std::vector<T2> &b) {
        using T3 = decltype(std::declval<T1>() * std::declval<T2>());
        if (a.getColumnSize() != b.size()) {
            throw ArgumentNotMatchException("Matrix and Vector size not match");
        }

        Matrix<T3> c(a.getRowSize(), 1);

        for (int i = 0; i < a.getRowSize(); ++i) {
            for (int j = 0; j < a.getColumnSize(); ++j) {
                c.Access(i, 1) = c.Access(i, 1) + a.Access(i, j) * b[j];
            }
        }
        return c;
    }

    /**
     * @brief operator overload, extra support added if two types can be operated.
     * @tparam T1
     * @tparam T2
     * @param a
     * @param b
     * @return
     */
    template<typename T1, typename T2>
    [[nodiscard]] auto operator/(Matrix<T1> &a, const T2 &b) {
        using T3 = decltype(std::declval<T1>() * std::declval<T2>());
        uint64_t n = a.getRowSize();
        uint64_t m = a.getColumnSize();
        Matrix<T3> c(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                c[i][j] = a.Access(i, j) / b;
            }
        }
    }

    /**
     * @warning It is very slow to print the matrix to the output
     * @tparam T
     * @param ostream
     * @param matrix
     * @return the reference of ostream itself
     */
    template<typename T>
    auto &operator<<(std::ostream &ostream, const Matrix<T> &matrix) {
        for (int i = 0; i < matrix.getRowSize(); ++i) {
            for (int j = 0; j < matrix.getColumnSize(); ++j) {
                ostream << matrix[i][j] << ' ';
            }
            ostream << '\n';
        }
        return ostream;
    }

    /**
     * Implement the transpose of a Matrix
     * @tparam T
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::transpose() {
        Matrix<T> res(this->column_size_, this->row_size_);
        for (uint64_t i = 0; i < this->row_size_; ++i) {
            for (uint64_t j = 0; j < this->column_size_; ++j) {
                res.Access(j, i) = this->Access(i, j);
            }
        }
        return res;
    }

    /**
     * @static get a identity Matrix of given size
     * @tparam T
     * @param s
     * @param t
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::identity(uint64_t s, T t) {
        Matrix<T> res(s, s);
        for (uint64_t i = 0; i < s; ++i) {
            res.Access(i, i) = t;
        }
        return res;
    }

    /**
     * @brief
     * @tparam T
     * @param col
     * @param ele
     * @return
     */
    template<typename T>
    Matrix<double> Matrix<T>::Householder(uint64_t col, uint64_t ele) {
        double square = 0;
        for (uint64_t i = ele - 1; i < row_size_ * column_size_; ++i) {
            square += pow(this->Access(i, col - 1), 2);
        }
        double mod = this->Access(ele - 1, col - 1) > 0 ? pow(square, 0.5) : -pow(square, 0.5);
        double modulus = mod * (mod + this->Access(ele - 1, col - 1));
        Matrix<double> U(this->row_size_, 1);
        for (int j = 0; j < this->row_size_; ++j) {
            if (j > ele - 1) {
                U.Access(j, 0) = this->Access(j, col - 1);
            } else {
                U.Access(j, 0) = 0;
            }
        }
        U.Access(ele - 1, 0) = this->Access(ele - 1, col - 1) + mod;
        return Matrix<double>::identity(this->row_size_, 1) - U * U.transpose() / modulus;
    }

    /**
     * @brief
     * @tparam T
     * @param col
     * @param ele
     * @return
     */
    template<typename T>
    Matrix<double> Matrix<T>::Hessenberg() {
        if (this->column_size_ != this->row_size_) {
            throw std::invalid_argument("The matrix needs to be square");
        }
        Matrix<double> left_H = Matrix<double>::identity(this->row_size_, 1);
        Matrix<double> right_H = Matrix<double>::identity(this->row_size_, 1);
        Matrix<double> H = left_H * (*this) * right_H;
        for (int i = 1; i < this->row_size_ - 1; ++i) {
            if (abs(H.Access(i + 1, i - 1)) > eps) {
                left_H = left_H * H.Householder(i, i + 1);
                right_H = H.Householder(i, i + 1) * right_H;
                H = left_H * (*this) * right_H;
            }
        }
        return H;
    }

    /**
     *
     * @tparam T
     * @param col
     * @param begin
     * @param end
     * @return
     */
    template<typename T>
    Matrix<double> Matrix<T>::Givens(uint64_t col, uint64_t begin, uint64_t end) {
        Matrix<double> R = Matrix<double>::identity(this->row_size_, 1);
        double r = pow(pow(this->Access(begin - 1, col - 1), 2) + pow(this->Access(end - 1, col - 1), 2), 0.5);
        double c = 1;
        double s = 0;
        if (abs(r) > eps) {
            c = this->Access(begin - 1, col - 1) / r;
            s = this->Access(end - 1, col - 1) / r;
        }
        R.Access(begin - 1, begin - 1) = c;
        R.Access(begin - 1, end - 1) = s;
        R.Access(end - 1, begin - 1) = -s;
        R.Access(end - 1, end - 1) = c;
        return R;
    }

    /**
     * @brief QR iteration implemented
     * @tparam T
     * @return
     */
    template<typename T>
    Matrix<double> Matrix<T>::QR_iteration() {
        Matrix<double> R = this->Hessenberg();
        Matrix<double> Q = Matrix<double>::identity(this->row_size_, 1);
        for (uint64_t i = 1; i < this->row_size_; ++i) {
            Matrix<double> temp_R = R.Givens(i, i, i + 1);
            R = temp_R * R;
            Q = Q * temp_R.transpose();
        }
        return R * Q;
    }

    /**
     * @brief get eigen value implemented
     * @tparam T
     * @return
     */
    template<typename T>
    std::vector<double> Matrix<T>::eigenvalue() {
        if (this->column_size_ != this->row_size_) {
            throw std::invalid_argument("The matrix needs to be square");
        }
        std::vector<double> eigenvalues(this->row_size_);
        static constexpr int32_t iter_times = 150;
        Matrix<double> H = this->QR_iteration();
        for (int i = 0; i < iter_times; ++i) {
            H = H.QR_iteration();
        }
        for (int j = 0; j < this->row_size_; ++j) {
            eigenvalues[j] = H.Access(j, j);
        }
        return eigenvalues;
    }

    /**
     * @brief Inverse of Matrix
     * @tparam T
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::inverse() {
        if ((this->column_size_ != this->row_size_) && this->determinant() == 0) {
            throw std::invalid_argument("The matrix does not meet the invertible condition (square matrix and the determinant is not 0)");
        }
        Matrix<T> res(this->row_size_, this->column_size_, static_cast<T>(0));
        for (int32_t i = 0; i < this->row_size_; i++) {
            for (int32_t j = 0; j < this->column_size_; j++) {
                std::vector< std::vector <T>> submatrix(this->row_size_,std::vector<T>(this->column_size_, static_cast<T>(0)));
                for(int r=0;r<this->row_size_;r++){
                    for(int c=0;c<this->column_size_;c++){
                        submatrix[r][c]=this->Access(r,c);
                    }
                }
                submatrix.erase(submatrix.begin() + i);
                for (int m = 0; m < this->row_size_ - 1; ++m) {
                    submatrix[m].erase(submatrix[m].begin() + j);
                }
                Matrix<T> sub(submatrix.size(),submatrix[0].size());
                for(int r=0;r<submatrix.size();r++){
                    for(int c=0;c<submatrix[0].size();c++){
                        sub.Access(r,c)=submatrix[r][c];
                    }
                }
                res[j][i] =
                        (((i + j) % 2) ? -1 : 1) * sub.determinant();
            }
        }
        Matrix<T> ans = Matrix<T>{res};
        ans = ans / this->determinant();
        return ans;
    }

    /**
     * @brief trace of Matrix
     * @tparam T
     * @return
     */
    template<typename T>
    T Matrix<T>::trace() {
        T ans{0};
        if (this->column_size_ != this->row_size_) {
            throw std::invalid_argument("The matrix needs to be square");
        }
        for (int32_t i = 0; i < this->row_size_; ++i) {
            ans += this->Access(i, i);
        }
        return ans;
    }

    /**
     * @brief determinant of Matrix
     * @tparam T
     * @return
     */
    template<typename T>
    T Matrix<T>::determinant() {
        if (this->column_size_ != this->row_size_) {
            throw std::invalid_argument("The matrix needs to be square");
        }
        uint64_t size_m = this->row_size_;
        if (size_m == 1) {
            return this->Access(0, 0);
        }
        Matrix<T> submatrix(size_m - 1, size_m - 1, 0);
        T ans(0);
        for (uint32_t i = 0; i < size_m; ++i) {
            for (uint32_t j = 0; j < size_m - 1; ++j) {
                for (uint32_t k = 0; k < size_m - 1; ++k) {
                    submatrix.Access(j, k) = this->Access((((i > j) ? 0 : 1) + j), k + 1);
                }
            }
            ans += ((i % 2) ? -1 : 1) * this->Access(i, 0) * submatrix.determinant();
        }
        return ans;
    }

    /**
     * @brief reshape the Matrix of the given size
     * @tparam T
     * @param row
     * @param col
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::reshape(int32_t row, int32_t col) {
        int32_t col_num = this->column_size_;
        int32_t num = this->row_size_ * col_num;
        if (row * col != num || num <= 0) {
            throw std::invalid_argument("The matrix needs to be square");
            return *this;
        }
        Matrix<T> res(row, col);
        for (int i = 0; i < num; i++) {
            res.Access(i / col, i % col) = this->Access(i / col_num, i % col_num);
        }
        return Matrix<T>(std::move(res));
    }

    /**
     * @brief slice the Matrix into a smaller one
     * @tparam T
     * @param row1
     * @param row2
     * @param col1
     * @param col2
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::slice(int32_t row1, int32_t row2, int32_t col1, int32_t col2) {
        if (row1 < 0 || row2 >= this->row_size_ || col1 < 0 || col2 >= this->column_size_ || row1 > row2 ||
            col1 > col2) {
            return *this;
        }
        Matrix<T> res(row2 - row1, col2 - col1);
        for (int32_t i = row1; i < row2; i++) {
            for (int32_t j = col1; j < col2; j++) {
                res.Access(i - row1, j - col1) = this->Access(i, j);
            }
        }
        return Matrix<T>(std::move(res));
    }

    /**
     * @brief get the conjugate of Matrix
     * @tparam T
     * @param conjugate
     * @return
     */
    template<typename T>
    Matrix<T> Matrix<T>::Conjugate(std::function<T(const T &)> conjugate) {
        auto result = Matrix<T>(row_size_, column_size_);
        int n = row_size_ * column_size_;
        for (int i = 0; i < n; ++i) {
            result.data_[i] = conjugate(data_[i]);
        }
        return result;
    }

    /**
     * @brief Convert cv::Mat to Matrix
     * @param mat
     * @return
     */
    Matrix<double> CvMatToMatrix(const cv::Mat &mat) {
        int32_t type = mat.type();
        if (mat.channels() > 1) {
            throw ArgumentNotMatchException("Channel not supported!");
        }
        if (mat.rows == 0 || mat.cols == 0) {
            throw BadSizeException("Size exception");
        }

        Matrix<double> result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                result[i][j] = mat.at<int>(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Convert Matrix to cv::Mat
     * @tparam T
     * @param matrix
     * @return
     */
    template<typename T>
    cv::Mat MatrixToCvMat(const Matrix<T> &matrix) {
        using std::is_same_v;
        int type = 7;
        if (is_same_v<T, uint8_t>()) {
            type = 0;
        } else if (is_same_v<T, int8_t>()) {
            type = 1;
        } else if (is_same_v<T, uint16_t>()) {
            type = 2;
        } else if (is_same_v<T, int16_t>()) {
            type = 3;
        } else if (is_same_v<T, int32_t>()) {
            type = 4;
        } else if (is_same_v<T, float>()) {
            type = 5;
        } else if (is_same_v<T, double>()) {
            type = 6;
        }

        if (type == 7) {
            throw ArgumentNotMatchException("Type not supported");
        }

        if (type < 5) {
            cv::Mat mat(matrix.getRowSize(), matrix.getColumnSize(), CV_32S);
            for (int i = 0; i < matrix.getRowSize(); ++i) {
                for (int j = 0; j < matrix.getColumnSize(); ++j) {
                    mat.at<T>(i, j) = matrix[i][j];
                }
            }
            return mat;
        } else {
            cv::Mat mat(matrix.getRowSize(), matrix.getColumnSize(), CV_64F);
            for (int i = 0; i < matrix.getRowSize(); ++i) {
                for (int j = 0; j < matrix.getColumnSize(); ++j) {
                    mat.at<T>(i, j) = matrix[i][j];
                }
            }
            return mat;
        }

    }

    /**
     * @brief Implement the eigen vector by using the eigen library
     * @tparam T
     * @return
     */
    template<typename T>
    std::vector<std::pair<T, std::vector<T>>> Matrix<T>::eigen() {
        auto mat = (cv::Mat_<T>)(*this);
        cv::Mat_<T> valuesMat;
        cv::Mat_<T> vectorsMat;

        cv::eigen(mat, valuesMat, vectorsMat);

        std::vector<std::pair<T, std::vector<T>>> result;
        for (int i = 0; i < valuesMat.rows; ++i) {
            std::vector<T> vec;
            for (int j = 0; j < vectorsMat.cols; ++j) {
                vec.push_back(vectorsMat[i][0]);
            }
            result.push_back(std::make_pair(valuesMat[i][0], vec));
        }
        return result;
    }

    /**
     * Cast operator overload (from Matrix to cv::Mat_ )
     * @tparam T
     * @return
     */
    template<typename T>
    Matrix<T>::operator cv::Mat_<T>() {
        cv::Mat_<T> mat(row_size_, column_size_);
#pragma omp parallel for collapse(2)
        for (int row = 0; row < row_size_; ++row) {
            for (int col = 0; col < column_size_; ++col) {
                mat(row, col) = data_[row * column_size_ + col];
            }
        }
        return mat;
    }
}

#endif //CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H
