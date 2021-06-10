//
// Created by GuoYubin on 2021/6/8.
//

#pragma GCC optimize(3)

#include "simple_matrix.h"
#include <iostream>
//#include <memory>
using namespace simple_matrix;
using std::cout;
int main() {
    clock_t start = clock();


//    Matrix<double> m1(10000, 10000);
//    Matrix<int> m2(10000, 10000, 5);
//    Matrix m4 = m1 + m2;

    Matrix<int> m3(100, 100);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            m3[i][j] = i * j;
        }
    }
    cout << m3.FindMax();
    cout << '\n';
    cout << m3.FindMin();
    cout << '\n';



    clock_t end = clock();
    cout << end - start;
    
}

