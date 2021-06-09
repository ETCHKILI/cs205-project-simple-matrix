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

#include <cstdint>
#include "simple_matrix_exception.h"
#include <iterator>

namespace simple_matrix {
    static const constexpr int64_t kDefaultRowSize = 10;
    static const constexpr int64_t kDefaultColumnSize = 10;
    static const constexpr int64_t kDefaultSideLength = 10;
    static const int64_t kMaxAllocateSize = UINT64_MAX;

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
        T* operator[](int row);

    protected:
        T *data_;
        uint64_t row_size_;
        uint64_t column_size_;
        explicit Matrix();
        void setRowSize(uint64_t rowSize);
        void setColumnSize(uint64_t columnSize);

    public:
        explicit Matrix(uint64_t row_size, uint64_t column_size);
        explicit Matrix(uint64_t row_size, uint64_t column_size, T initial_value);
        virtual ~Matrix();
        virtual T &Access(uint64_t row, uint64_t column);
        virtual void SetValue(T val);
        uint64_t getRowSize() const;
        uint64_t getColumnSize() const;
        static bool CheckSizeValid(uint64_t row_size, uint64_t column_size);
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
        data_ = nullptr;
        row_size_ = 0;
        column_size_ = 0;
    }

    /**
     * Constructor that do not initialize the data
     *
     * @tparam T
     * @param row_size
     * @param column_size
     */
    template<typename T>
    Matrix<T>::Matrix(uint64_t row_size, uint64_t column_size) {
        if ( !CheckSizeValid(row_size, column_size)) {
            throw simple_matrix::BadSizeException("Size too large!");
        }
        data_ = new T[row_size * column_size];
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
    Matrix<T>::Matrix(uint64_t row_size, uint64_t column_size, T initial_value) {
        if ( !CheckSizeValid(row_size, column_size)) {
            throw simple_matrix::BadSizeException("Size too large!");
        }
        data_ = new T[row_size * column_size];
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
    bool Matrix<T>::CheckSizeValid(uint64_t row_size, uint64_t column_size) {
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
    T &Matrix<T>::Access(uint64_t row, uint64_t column) {
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
    uint64_t Matrix<T>::getRowSize() const {
        return row_size_;
    }

    /**
     *
     * @tparam T
     * @return
     */
    template<typename T>
    uint64_t Matrix<T>::getColumnSize() const {
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
    void Matrix<T>::setRowSize(uint64_t rowSize) {
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
    void Matrix<T>::setColumnSize(uint64_t columnSize) {
        column_size_ = columnSize;
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
    class SymmetricMatrix : public Matrix<T>{
    private:

    public:
        explicit SymmetricMatrix();
        explicit SymmetricMatrix(uint64_t side_length);
        explicit SymmetricMatrix(uint64_t side_length, T initial_value);
        ~SymmetricMatrix();
        static bool CheckSizeValid(uint64_t side_length);
    };

    template<typename T>
    SymmetricMatrix<T>::SymmetricMatrix() {
        this->data_ = new T[kDefaultSideLength * (kDefaultSideLength - 1) / 2];
        this->setColumnSize(kDefaultSideLength);
        this->setRowSize(kDefaultSideLength);
    }

    template<typename T>
    SymmetricMatrix<T>::SymmetricMatrix(uint64_t side_length) {
        if (!CheckSizeValid(side_length)) {
            throw simple_matrix::BadSizeException("Size too large");
        }
        this->data_ = new T[side_length * (side_length - 1) / 2];
        this->setColumnSize(side_length);
        this->setRowSize(side_length);
    }

    template<typename T>
    SymmetricMatrix<T>::SymmetricMatrix(uint64_t side_length, T initial_value) {
        if (!CheckSizeValid(side_length)) {
            throw simple_matrix::BadSizeException("Size too large");
        }
        this->data_ = new T[side_length * (side_length - 1) / 2] {initial_value};
        this->setColumnSize(side_length);
        this->setRowSize(side_length);
    }

    template<typename T>
    bool SymmetricMatrix<T>::CheckSizeValid(uint64_t side_length) {
        return side_length <= kMaxAllocateSize / side_length && side_length > 0;
    }

    template<typename T>
    SymmetricMatrix<T>::~SymmetricMatrix() {
    }
}

#endif //CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H
