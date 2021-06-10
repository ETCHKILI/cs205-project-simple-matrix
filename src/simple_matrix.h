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
 * @Github https://github.com/ETCHKILI/cs205-project-simple-matrix
 * @Organization SUSTech
 * @Author Guo Yubin, Shen Zhilong, Fan Leilong
 */
#ifndef CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H
#define CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H

#include "simple_matrix_exception.h"

#include <cstdint>
#include <iterator>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace simple_matrix {

    static const constexpr int64_t kDefaultRowSize = 10;
    static const constexpr int64_t kDefaultColumnSize = 10;
    static const constexpr int64_t kDefaultSideLength = 10;
    static const uint64_t kMaxAllocateSize = UINT64_MAX;
    static const int64_t kOSBits = sizeof (void *) * 8;

    using local_uint_t = uint64_t;

    enum class SelectAs {
        SUB_MATRIX, ROW, COLUMN
    };


/**
     *
     * @class Matrix
     * @brief Base class in simple_matrix
     * @tparam T
     * @details Use one-dimension array to simulate a matrix
     */
    template<typename T>
    class Matrix {
    private:

    protected:
        T *data_;
        local_uint_t row_size_;
        local_uint_t column_size_;
        explicit Matrix();
        void setRowSize(local_uint_t rowSize);
        void setColumnSize(local_uint_t columnSize);

    public:
        T* operator[](int row);

        explicit Matrix(local_uint_t row_size, local_uint_t column_size);
        explicit Matrix(local_uint_t row_size, local_uint_t column_size, const T& initial_value);
        explicit Matrix(const std::vector<T>& v, SelectAs selectAs);

        Matrix(Matrix<T>& that);
        Matrix(Matrix<T>&& that) noexcept;

        Matrix<T> &operator=(Matrix<T> &that);
        Matrix<T> &operator=(Matrix<T> &&that) noexcept;

        virtual ~Matrix();
        virtual T &Access(local_uint_t row, local_uint_t column);
        virtual void SetValue(T val);
        local_uint_t getRowSize() const;
        local_uint_t getColumnSize() const;
        static bool CheckSizeValid(local_uint_t row_size, local_uint_t column_size);

        [[nodiscard]] Matrix<T> operator+(Matrix<T> &that);
        [[nodiscard]] Matrix<T> operator-(Matrix<T> &that);
        [[nodiscard]] Matrix<T> operator*(Matrix<T> &that);
        void operator *=(T k);

        T FindMin(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX, std::function<bool(const T&, const T&)> compare = [](const T&a, const T&b){return a < b;});
        T FindMax(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX, std::function<bool(const T&, const T&)> compare = [](const T&a, const T&b){return a > b;});
        T Sum(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX);
        T Average(std::vector<int32_t> index = std::vector<int32_t>(), SelectAs selectAs = SelectAs::SUB_MATRIX);

        Matrix <T> getRotate180();

        Matrix <T> subMatrix(uint64_t row_lo, uint64_t col_lo, uint64_t row_hi, uint64_t col_hi);

        Matrix <T> convolution(Matrix <T> &that);
    };

    /**
     * @brief
     * @details
     * @tparam T
     * @param
     * @attention
     * @warning IT IS PROTECTED!!! Should only be used when you want to initialize the data_ as nullptr
     */
    template<typename T>
    Matrix<T>::Matrix() {
        data_ = new T[1]();
        row_size_ = 1;
        column_size_ = 1;
    }

    /**
     * Constructor that do not initialize the data
     *
     * @tparam T
     * @param row_size
     * @param column_size
     */
    template<typename T>
    Matrix<T>::Matrix(local_uint_t row_size, local_uint_t column_size) {
        if ( !CheckSizeValid(row_size, column_size)) {
            throw simple_matrix::BadSizeException("Size too large!");
        }
        data_ = new T[row_size * column_size]();
        column_size_ = column_size;
        row_size_ = row_size;
    }

    /*!
     * Constructor that initialize the data
     *
     * @tparam T
     * @param row_size
     * @param column_size
     * @param initial_value
     */
    template<typename T>
    Matrix<T>::Matrix(local_uint_t row_size, local_uint_t column_size, const T& initial_value) {
        if ( !CheckSizeValid(row_size, column_size)) {
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
    *
    * @tparam T
    * @param row_size
    * @param column_size
    * @param initial_value
    */
    template<typename T>
    Matrix<T>::~Matrix() {
        if (data_ != nullptr) {
            delete []data_;
        }
    }

    /**
     * @protected by simple_matrix::Matrix
     * @warning THIS IS PROTECTED. Use Access(int, int) to access the
     * data unless you are writing a derived class and want to access
     * the data directly.
     * @tparam T
     * @param row
     * @return
     */
    template<typename T>
    T *Matrix<T>::operator[](int row) {
        return data_ + row * column_size_;
    }

    /**
     *
     * @tparam T
     * @param row_size
     * @param column_size
     * @return
     */
    template<typename T>
    bool Matrix<T>::CheckSizeValid(local_uint_t row_size, local_uint_t column_size) {
        return ((row_size <= kMaxAllocateSize / column_size) || (column_size <= kMaxAllocateSize / row_size)) && std::max(row_size, column_size) > 0;
    }

    /**
     *
     * @tparam T
     * @param val
     */
    template<typename T>
    void Matrix<T>::SetValue(T val) {
        /// TODO handle this exception
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
     *
     * @tparam T
     * @param row
     * @param column
     * @return
     */
    template<typename T>
    T &Matrix<T>::Access(local_uint_t row, local_uint_t column) {
        if (row > row_size_ - 1 || column > column_size_ - 1) {
            throw BadAccessException("Index out of bounds");
        }
        return (*this)[row][column];
    }

    /**
     *
     * @tparam T
     * @return
     */
    template<typename T>
    local_uint_t Matrix<T>::getRowSize() const {
        return row_size_;
    }

    /**
     *
     * @tparam T
     * @return
     */
    template<typename T>
    local_uint_t Matrix<T>::getColumnSize() const {
        return column_size_;
    }

    /**
     * @protected by Matrix, you should only use this when
     * you want to initialize the data of matrix IN YOUR DERIVED
     * CLASS and have to set size for initializing matrix
     *
     * @tparam T
     * @param rowSize
     */
    template<typename T>
    void Matrix<T>::setRowSize(local_uint_t rowSize) {
        row_size_ = rowSize;
    }

    /**
     * @protected by Matrix, you should only use this when
     * you want to initialize the data of matrix IN YOUR DERIVED
     * CLASS and have to set size for initializing matrix
     *
     * @tparam T
     * @param columnSize
     */
    template<typename T>
    void Matrix<T>::setColumnSize(local_uint_t columnSize) {
        column_size_ = columnSize;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator+(Matrix<T> &that) {
        if (this->column_size_ != that.column_size_ || this->row_size_ != that.row_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T> result(this->row_size_, this->column_size_);
        for (int i = 0; i < this->row_size_; ++i) {
            for (int j = 0; j < this->column_size_; ++j) {
                result.Access(i, j) = Access(i, j) + that.Access(i, j);
            }
        }
        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator-(Matrix<T> &that) {
        if (this->column_size_ != that.column_size_ || this->row_size_ != that.row_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        Matrix<T> result(this->row_size_, this->column_size_);
        for (int i = 0; i < this->row_size_; ++i) {
            for (int j = 0; j < this->column_size_; ++j) {
                result.Access(i, j) = Access(i, j) - that.Access(i, j);
            }
        }
        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator*(Matrix<T> &that) {
        if (this->column_size_ != that.row_size_ || this->row_size_ != that.column_size_) {
            throw ArgumentNotMatchException("Matrix size not match");
        }
        int n = this->row_size_;
        Matrix<T> result(n, n);
        for (int i = 0; i < n; ++i) {
            for (int k = 0;k < this->column_size_; ++k) {
                T temp = Access(i , k);
                for (int j = 0; j < n; ++j) {
                    result.Access(i, j) = result.Access(i, j) + temp * that.Access(k, j);
                }
            }
        }
        return result;
    }

    template<typename T>
    void Matrix<T>::operator*=(T k) {
        auto new_data = new T[row_size_ * column_size_];
        int n = row_size_ * column_size_;
        for (int i = 0; i < n; ++i) {
            new_data[i] = data_[i] * k;
        }
        delete[] this->data_;
        this->data_ = new_data;
    }

    template<typename T>
    Matrix<T>::Matrix(const std::vector<T>& v, SelectAs selectAs) {
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

    template<typename T>
    Matrix<T>::Matrix(Matrix<T> &that) {
        data_ = new T[that.row_size_ * that.column_size_];
        row_size_ = that.row_size_;
        column_size_ = that.column_size_;
        for (int i = 0; i < row_size_; ++i) {
            for (int j = 0; j < column_size_; ++j) {
                data_[i * column_size_ + j] = that.data_[i * column_size_ + j];
            }
        }
    }

    template<typename T>
    Matrix<T>::Matrix(Matrix<T> &&that) noexcept{
        data_ = that.data_;
        row_size_ = that.row_size_;
        column_size_ = that.column_size_;
        that.data_ = nullptr;
    }

    template<typename T>
    Matrix<T> &Matrix<T>::operator=(Matrix<T> &that) {
        if (this != that) {
            data_ = new T[that.row_size_ * that.column_size_];
            row_size_ = that.row_size_;
            column_size_ = that.column_size_;
            for (int i = 0; i < row_size_; ++i) {
                for (int j = 0; j < column_size_; ++j) {
                    data_[i * column_size_ + j] = that.data_[i * column_size_ + j];
                }
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T> &Matrix<T>::operator=(Matrix<T> &&that) noexcept{
        if (this != that) {
            data_ = that.data_;
            row_size_ = that.row_size_;
            column_size_ = that.column_size_;
            that.data_ = nullptr;
        }
        return *this;
    }

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
            for (int k = 0;k < ma; ++k) {
                T1 temp = a.Access(i , k);
                for (int j = 0; j < na; ++j) {
                    c.Access(i, j) = c.Access(i, j) + temp * b.Access(k, j);
                }
            }
        }
        return c;
    }

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

    template<typename T1, typename T2>
    [[nodiscard]] auto operator/(Matrix<T1> &a,const T2 &b) {
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
     * @deprecated Never use this and just take it
     * as a function for testing
     * @tparam T
     * @param ostream
     * @param matrix
     * @return the reference of ostream itself
     */
    template<typename T>
    auto &operator<<(std::ostream& ostream, Matrix<T>& matrix) {
        for (int i = 0; i < matrix.getRowSize(); ++i) {
            for (int j = 0; j < matrix.getColumnSize(); ++j) {
                ostream << matrix.Access(i, j);
            }
            ostream << '\n';
        }
        return ostream;
    }

    template<typename T>
    Matrix<T> Matrix<T>::convolution(Matrix<T> &that) {
        uint64_t new_row_size = this->row_size_ + that.row_size_ - 1;
        uint64_t new_col_size = this->column_size_ + that.column_size_ - 1;
        Matrix<T> res (new_row_size, new_col_size);
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

    template<typename T>
    Matrix<T> Matrix<T>::subMatrix(uint64_t row_lo, uint64_t col_lo, uint64_t row_hi, uint64_t col_hi) {
        uint64_t new_row_size = row_hi - row_lo + 1;
        uint64_t new_col_size = col_hi - col_lo + 1;
        Matrix<T> res(new_row_size, new_col_size);
        for (int i = 0; i < new_row_size; ++i) {
            for (int j = 0; j < new_col_size; ++j) {
                res.Access(i,j) = Access(row_lo + i,col_lo + j);
            }
        }
        return res;
    }

    template<typename T>
    Matrix<T> Matrix<T>::getRotate180() {
        Matrix<T> res(row_size_, column_size_);
        for (int i = 0; i < row_size_ * column_size_; ++i) {
            res.Access(i/column_size_,i %column_size_) = Access(row_size_-1-(int)(i/column_size_),column_size_-1-i%column_size_);
        }
        return res;
    }

    template<typename T>
    T Matrix<T>::FindMin(std::vector<int32_t> index, SelectAs selectAs, std::function<bool(const T&, const T&)> compare) {
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
                index.push_back(row_size_ - 1);
                index.push_back(column_size_ - 1);
            }
            res = data_[index[0] * column_size_ + index[1]];
            for (int i = index[0]; i < index[2]; ++i) {
                for (int j = index[1]; j < index[3]; ++j) {
                    if (compare(data_[i * column_size_ + j], res)){
                        res = data_[i * column_size_ + j];
                    }
                }
            }
        }
        return res;
    }

    template<typename T>
    T Matrix<T>::FindMax(std::vector<int32_t> index, SelectAs selectAs, std::function<bool(const T&, const T&)> compare) {
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
                index.push_back(row_size_ - 1);
                index.push_back(column_size_ - 1);
            }
            res = data_[index[0] * column_size_ + index[1]];
            for (int i = index[0]; i < index[2]; ++i) {
                for (int j = index[1]; j < index[3]; ++j) {
                    if (compare(data_[i * column_size_ + j], res)){
                        res = data_[i * column_size_ + j];
                    }
                }
            }
        }
        return res;
    }

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
                index.push_back(row_size_ - 1);
                index.push_back(column_size_ - 1);
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

    template<typename T>
    T Matrix<T>::Average(std::vector<int32_t> index, SelectAs selectAs) {
        return (Sum(index, selectAs) / (row_size_ * column_size_));
    }


    /**
     * TODO complete it if we still have time :)
     * @class SymmetricMatrix
     * @brief Base class in simple_matrix
     * @tparam T
     * @details Use one-dimension array to simulate a matrix
     *          Considering it is symmetric, save data[i][j] in the same space as data[j][i]
     */
    template<typename T>
    class SymmetricMatrix : public Matrix<T> {
    private:

    public:
        explicit SymmetricMatrix();

        explicit SymmetricMatrix(local_uint_t side_length);

        explicit SymmetricMatrix(local_uint_t side_length, T initial_value);

        ~SymmetricMatrix();

        static bool CheckSizeValid(local_uint_t side_length);
    };

}

#endif //CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H
