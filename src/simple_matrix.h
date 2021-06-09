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
#include <iterator>
#include <vector>
using namespace std;

namespace simple_matrix
{
    static const constexpr int64_t kDefaultRowSize = 10;
    static const constexpr int64_t kDefaultColumnSize = 10;
    static const constexpr int64_t kDefaultSideLength = 10;
    static const int64_t kMaxAllocateSize = UINT64_MAX;
    static const double eps = 1e-3;

    /**
     *
     * @class Matrix
     * @brief Base class in simple_matrix
     * @tparam T
     * @details Use one-dimension array to simulate a matrix
     */
    template <typename T>
    class Matrix
    {
    private:
        T *operator[](int row);

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
        Matrix<T> Matrix<T>::identity(uint64_t s, T t);
        Matrix<double> Householder(uint64_t col, uint64_t ele) const;
        Matrix<double> Matrix<T>::Hessenberg() const;
        Matrix<double> Matrix<T>::Givens(uint64_t col, uint64_t begin, uint64_t end) const;
        Matrix<double> Matrix<T>::QR_iteration() const;
        vector<double> Matrix<T>::eigenvalue() const;
        Matrix<double> Matrix<T>::eigenvector() const;
        T Matrix<T>::trace() const;
        T Matrix<T>::determinant() const;
        Matrix<T> Matrix<T>::reshape(int32_t row, int32_t col) const;
        Matrix<T> Matrix<T>::slice(int32_t row1, int32_t row2, int32_t col1, int32_t col2) const;
    };

    /**
     * @brief
     * @details
     * @tparam T
     * @param
     * @attention
     * @warning IT IS PROTECTED!!! Should only be used when you want to initialize the data_ as nullptr
     */
    template <typename T>
    Matrix<T>::Matrix()
    {
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
    template <typename T>
    Matrix<T>::Matrix(uint64_t row_size, uint64_t column_size)
    {
        if (!CheckSizeValid(row_size, column_size))
        {
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
    template <typename T>
    Matrix<T>::Matrix(uint64_t row_size, uint64_t column_size, T initial_value)
    {
        if (!CheckSizeValid(row_size, column_size))
        {
            throw simple_matrix::BadSizeException("Size too large!");
        }
        data_ = new T[row_size * column_size];
        column_size_ = column_size;
        row_size_ = row_size;
        for (int i = 0; i < row_size_; ++i)
        {
            for (int j = 0; j < column_size_; ++j)
            {
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
    template <typename T>
    Matrix<T>::~Matrix()
    {
        if (data_ != nullptr)
        {
            delete[] data_;
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
    template <typename T>
    T *Matrix<T>::operator[](int row)
    {
        return data_ + row * column_size_;
    }

    /**
     *
     * @tparam T
     * @param row_size
     * @param column_size
     * @return
     */
    template <typename T>
    bool Matrix<T>::CheckSizeValid(uint64_t row_size, uint64_t column_size)
    {
        return ((row_size <= kMaxAllocateSize / column_size) || (column_size <= kMaxAllocateSize / row_size)) && std::max(row_size, column_size) > 0;
    }

    /**
     *
     * @tparam T
     * @param val
     */
    template <typename T>
    void Matrix<T>::SetValue(T val)
    {
        /// TODO handle this exception
        if (data_ == nullptr)
        {
            throw BadAccessException("Matrix is not initialized or index out of range");
        }
        for (int i = 0; i < row_size_; ++i)
        {
            for (int j = 0; j < column_size_; ++j)
            {
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
    template <typename T>
    T &Matrix<T>::Access(uint64_t row, uint64_t column)
    {
        if (row > row_size_ - 1 || column > column_size_ - 1)
        {
            throw BadAccessException("Index out of bounds");
        }
        return (*this)[row][column];
    }

    /**
     *
     * @tparam T
     * @return
     */
    template <typename T>
    uint64_t Matrix<T>::getRowSize() const
    {
        return row_size_;
    }

    /**
     *
     * @tparam T
     * @return
     */
    template <typename T>
    uint64_t Matrix<T>::getColumnSize() const
    {
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
    template <typename T>
    void Matrix<T>::setRowSize(uint64_t rowSize)
    {
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
    template <typename T>
    void Matrix<T>::setColumnSize(uint64_t columnSize)
    {
        column_size_ = columnSize;
    }

    template <typename T>
    Matrix<T> Matrix<T>::identity(uint64_t s, T t)
    {
        Matrix<T> res(s, s);
        for (uint64_t i = 0; i < s; ++i)
        {
            res.Access(i, i) = t;
        }
        return res;
    }

    template <typename T>
    Matrix<double> Matrix<T>::Householder(uint64_t col, uint64_t ele) const
    {
        double square = 0;
        for (uint64_t i = ele - 1; i < row_size_ * column_size_; ++i)
        {
            square += pow(this->Access(i, col - 1), 2);
        }
        double mod = this->Access(ele - 1, col - 1) > 0 ? pow(square, 0.5) : -pow(square, 0.5);
        double modulus = mod * (mod + this->Access(ele - 1, col - 1));
        Matrix<double> U;
        U.SetValue(1);
        for (int j = 0; j < this->row_size_; ++j)
        {
            if (j > ele - 1)
            {
                U.Access(j, 0) = this->Access(j, col - 1);
            }
            else
            {
                U.Access(j, 0) = 0;
            }
        }
        U.Access(ele - 1, 0) = this->Access(ele - 1, col - 1) + mod;
        return Matrix<double>::identity(this->row_size_, 1) - U * U.transpose() / modulus;
    }

    template <typename T>
    Matrix<double> Matrix<T>::Hessenberg() const
    {
        if (this->column_size_ != this->row_size_)
        {
            throw std::invalid_argument("The matrix needs to be square");
        }
        Matrix<double> left_H = Matrix<double>::identity(this->row_size_, 1);
        Matrix<double> right_H = Matrix<double>::identity(this->row_size_, 1);
        Matrix<double> H = left_H * this * right_H;
        for (int i = 1; i < this->row_size_ - 1; ++i)
        {
            if (abs(H.Access(i + 1, i - 1)) > eps)
            {
                left_H = left_H * H.Householder(i, i + 1);
                right_H = H.Householder(i, i + 1) * right_H;
                H = left_H * this * right_H;
            }
        }
        return H;
    }

    template <typename T>
    Matrix<double> Matrix<T>::Givens(uint64_t col, uint64_t begin, uint64_t end) const
    {
        Matrix<double> R = Matrix<double>::identity(this->row_size_, 1);
        double r = pow(pow(this->Access(begin - 1, col - 1), 2) + pow(this->Access(end - 1, col - 1), 2), 0.5);
        double c = 1;
        double s = 0;
        if (abs(r) > eps)
        {
            c = this->Access(begin - 1, col - 1) / r;
            s = this->Access(end - 1, col - 1) / r;
        }
        R.Access(begin - 1, begin - 1) = c;
        R.Access(begin - 1, end - 1) = s;
        R.Access(end - 1, begin - 1) = -s;
        R.Access(end - 1, end - 1) = c;
        return R;
    }

    template <typename T>
    Matrix<double> Matrix<T>::QR_iteration() const
    {
        Matrix<double> R = this->Hessenberg();
        Matrix<double> Q = Matrix<double>::identity((this->row_size_) * (this->column_size_), 1);
        for (uint64_t i = 1; i < (this->row_size_) * (this->column_size_); ++i)
        {
            Matrix<double> temp_R = R.Givens(i, i, i + 1);
            R = temp_R * R;
            Q = Q * temp_R.transpose();
        }
        return R * Q;
    }

    template <typename T>
    vector<double> Matrix<T>::eigenvalue() const
    {
        if (this->column_size_ != this->row_size_)
        {
            throw std::invalid_argument("The matrix needs to be square");
        }
        vector<double> eigenvalues(this->row_size_);
        static constexpr int32_t iter_times = 150;
        Matrix<double> H = this->QR_iteration();
        for (int i = 0; i < iter_times; ++i)
        {
            H = H.QR_iteration();
        }
        for (int j = 0; j < this->row_size_; ++j)
        {
            eigenvalues[j] = H.Access(j, j);
        }
        return eigenvalues;
    }

    template <typename T>
    T Matrix<T>::trace() const
    {
        T ans{0};
        if (this->column_size_ != this->row_size_)
        {
            throw std::invalid_argument("The matrix needs to be square");
        }
        for (int32_t i = 0; i < this->rows(); ++i)
        {
            ans += this->Access(i, i);
        }
        return ans;
    }

    template <typename T>
    T Matrix<T>::determinant() const
    {
        if (this->column_size_ != this->row_size_)
        {
            throw std::invalid_argument("The matrix needs to be square");
        }
        uint64_t size_m = this->column_size_ * this->row_size_;
        if (size_m == 1)
        {
            return this->Access(0, 0);
        }
        Matrix<T> submatrix(size_m - 1, size_m - 1, 0);
        T ans(0);
        for (uint32_t i = 0; i < size_m; ++i)
        {
            for (uint32_t j = 0; j < size_m - 1; ++j)
            {
                for (uint32_t k = 0; k < size_m - 1; ++k)
                {
                    submatrix.Access(j, k) = this->Access((((i > j) ? 0 : 1) + j), k + 1);
                }
            }
            ans += ((i % 2) ? -1 : 1) * this->Access(i, 0) * determinant(submatrix);
        }
        return ans;
    }

    template <typename T>
    Matrix<T> Matrix<T>::reshape(int32_t row, int32_t col) const
    {
        int32_t col_num = this->column_size_;
        int32_t num = this->row_size_ * col_num;
        if (row * col != num || num <= 0)
        {
            throw std::invalid_argument("The matrix needs to be square");
            return this;
        }
        Matrix<T> res;
        for (int i = 0; i < num; i++)
        {
            res.Access(i / col, i % col) = this->Access(i / col_num, i % col_num);
        }
        return Matrix<T>(std::move(res));
    }

    template <typename T>
    Matrix<T> Matrix<T>::slice(int32_t row1, int32_t row2, int32_t col1, int32_t col2) const
    {
        if (row1 < 0 || row2 >= this->row_size_ || col1 < 0 || col2 >= this->column_size_ || row1 > row2 || col1 > col2)
        {
            return this;
        }
        Matrix<T> res(row2 - row1 + 1, col2 - col1 + 1);
        for (int32_t i = row1; i < row2; i++)
        {
            for (int32_t j = col1; j < col2; j++)
            {
                res.Access(i - row1, j - col1) = this->Access(i, j);
            }
        }
        return Matrix<T>(std::move(res));
    }

    /*!
     * TODO complete it if we still have time :)
     * @class SymmetricMatrix
     * @brief Base class in simple_matrix
     * @tparam T
     * @details Use one-dimension array to simulate a matrix
     *          Considering it is symmetric, save data[i][j] in the same space as data[j][i]
     */
    template <typename T>
    class SymmetricMatrix : public Matrix<T>
    {
    private:
    public:
        explicit SymmetricMatrix();
        explicit SymmetricMatrix(uint64_t side_length);
        explicit SymmetricMatrix(uint64_t side_length, T initial_value);
        ~SymmetricMatrix();
        static bool CheckSizeValid(uint64_t side_length);
    };

    template <typename T>
    SymmetricMatrix<T>::SymmetricMatrix()
    {
        this->data_ = new T[kDefaultSideLength * (kDefaultSideLength - 1) / 2];
        this->setColumnSize(kDefaultSideLength);
        this->setRowSize(kDefaultSideLength);
    }

    template <typename T>
    SymmetricMatrix<T>::SymmetricMatrix(uint64_t side_length)
    {
        if (!CheckSizeValid(side_length))
        {
            throw simple_matrix::BadSizeException("Size too large");
        }
        this->data_ = new T[side_length * (side_length - 1) / 2];
        this->setColumnSize(side_length);
        this->setRowSize(side_length);
    }

    template <typename T>
    SymmetricMatrix<T>::SymmetricMatrix(uint64_t side_length, T initial_value)
    {
        if (!CheckSizeValid(side_length))
        {
            throw simple_matrix::BadSizeException("Size too large");
        }
        this->data_ = new T[side_length * (side_length - 1) / 2]{initial_value};
        this->setColumnSize(side_length);
        this->setRowSize(side_length);
        Vector<T>(1, 1, 1);
    }

    template <typename T>
    bool SymmetricMatrix<T>::CheckSizeValid(uint64_t side_length)
    {
        return side_length <= kMaxAllocateSize / side_length;
    }

    template <typename T>
    SymmetricMatrix<T>::~SymmetricMatrix()
    {
    }
}

#endif //CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_H
