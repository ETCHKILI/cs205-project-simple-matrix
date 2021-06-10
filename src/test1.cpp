//
// Created by GuoYubin on 2021/6/8.
//

#pragma GCC optimize(3)

#include "simple_matrix.h"
#include <iostream>
//#include <memory>
using namespace simple_matrix;
using std::cout;

void testeigenv() {
    using namespace simple_matrix;
    Matrix<std::complex<int>> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = std::complex(1, 2);
        }
    }
//    printMatrix(a);
    std::cout << a.Conjugate([] (std::complex<int> tmp) {return std::conj(tmp);});
};

int main() {
    clock_t start = clock();


//    Matrix<double> m1(10000, 10000);
//    Matrix<int> m2(10000, 10000, 5);
//    Matrix m4 = m1 + m2;



//    Matrix<int> m3(100, 100);
//    for (int i = 0; i < 100; ++i) {
//        for (int j = 0; j < 100; ++j) {
//            m3[i][j] = i * j;
//        }
//    }
//    cout << m3.FindMax();
//    cout << '\n';
//    cout << m3.FindMin();
//    cout << '\n';

    testeigenv();





    clock_t end = clock();
    cout << end - start;



}


