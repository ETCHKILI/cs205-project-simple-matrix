//
// Created by GuoYubin on 2021/6/8.
//

#include "simple_matrix.h"
#include <iostream>
#include <memory>

int main() {
    using namespace simple_matrix;

    Matrix A(2,2,0);
    Matrix B(2,2,0);
    A.Access(0,0)=1;
    A.Access(0,1)=2;
    A.Access(1,0)=3;
    A.Access(1,1)=4;
    B.Access(0,0)=-1;
    B.Access(0,1)=1;
    B.Access(1,0)=-2;
    B.Access(1,1)=2;

    auto res = A.convolution(B);
    B.getRotate180().print();
    res.print();

}

