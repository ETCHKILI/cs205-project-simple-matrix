//
// Created by GuoYubin on 2021/6/8.
//

#include "simple_matrix.h"
#include <iostream>
//#include <memory>

int main() {
    using namespace simple_matrix;

    Matrix matrix(3, 4, 5);
    auto m = Matrix(1,2,3);

    std::cout << matrix.Access(2, 3) << '\n';

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << matrix.Access(i, j) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    matrix.Access(1, 1) = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << matrix.Access(i, j) << ' ';
        }
        std::cout << '\n';
    }

    Matrix<int> m1(3,3);
    Matrix<int> m2(3,3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m1[i][j] = i + j;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m2[i][j] = i * j;
        }
    }
    Matrix m3 = m1 * m2;
    Matrix m4 = m1 + m2;
    std::cout << m1 * m2;

    m1.Access(0, 0) = 1;
    std::cout << m1.Access(0,0);



}

