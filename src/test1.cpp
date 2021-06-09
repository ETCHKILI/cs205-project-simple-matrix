//
// Created by GuoYubin on 2021/6/8.
//

#include "simple_matrix.h"
#include <iostream>
#include <memory>

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

    Matrix m1(3,3,3);
    Matrix m2(3,3,3);
    Matrix m3 = m1 * m2;
    Matrix m4 = m1 + m2;
    std::cout << m3 << m4;
}

