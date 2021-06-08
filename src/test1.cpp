//
// Created by GuoYubin on 2021/6/8.
//

#include "simple_matrix.h"
#include <iostream>

int main() {
    using namespace simple_matrix;

    Matrix<int> matrix(3, 4, 5);

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

}

