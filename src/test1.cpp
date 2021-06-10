#include <iostream>
#include <vector>
#include <complex>
#include "simple_matrix.h"

using namespace simple_matrix;

void testadd();

void testsub();

void testmul1();

void testmul2();

void testdot();

void testtrans();

void testconj();

void testMAX();

void testMIN();

void testsum();

void testtrace();

void testeigenv();

void testinv();

void testdet();

void testres();

void testsli();

void testconv();

template<typename T>
void printMatrix(Matrix<T> &mat) {
    for (int i = 0; i < mat.getRowSize(); ++i) {
        for (int j = 0; j < mat.getColumnSize(); ++j) {
            std::cout << mat.Access(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    testconv();
    return 0;
}

void testadd() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j;
        }
    }
    printMatrix(a);
    Matrix<int> b(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            b.Access(i, j) = i - j;
        }
    }
    printMatrix(b);
    Matrix<int> c(2, 2);
    c = a + b;
    printMatrix(c);
};

void testsub() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j;
        }
    }
    printMatrix(a);
    Matrix<int> b(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            b.Access(i, j) = i - j;
        }
    }
    printMatrix(b);
    Matrix<int> c(2, 2);
    c = a - b;
    printMatrix(c);
};

void testmul1() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j;
        }
    }
    printMatrix(a);
    Matrix<int> b(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            b.Access(i, j) = i - j;
        }
    }
    printMatrix(b);
    Matrix<int> c(2, 2);
    c = a * b;
    printMatrix(c);
};

void testmul2() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j;
        }
    }
    printMatrix(a);
    Matrix<int> c(2, 2);
    c = a * 5;
    printMatrix(c);
};

void testdot() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j;
        }
    }
    printMatrix(a);
    Matrix<int> b(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            b.Access(i, j) = i - j;
        }
    }
    printMatrix(b);
    Matrix<int> c(2, 2);
    c = DotMultiply(a, b);
    printMatrix(c);
};

void testconv() {
    Matrix<double> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2.0 + 1;
        }
    }
    printMatrix(a);

    Matrix<double> b(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            b.Access(i, j) = i - j * 2.0 + 1;
        }
    }
    printMatrix(b);

    Matrix<double> c(2, 2);
    c = a.convolution(b);
    printMatrix(c);
};

void testtrans() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2;
        }
    }
    printMatrix(a);
    Matrix<int> c(2, 2);
    c = a.transpose();
    printMatrix(c);
};

void testconj() {
    using namespace simple_matrix;
    Matrix<std::complex<int>> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = std::complex(1, 2);
        }
    }
    std::cout << a.Conjugate([] (std::complex<int> tmp) {return std::conj(tmp);});
};


void testMAX() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2 + 1;
        }
    }
    printMatrix(a);
    int ans;
    ans = a.FindMax();
    std::cout << ans << std::endl;
};

void testMIN() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2 + 1;
        }
    }
    printMatrix(a);
    int ans;
    ans = a.FindMin();
    std::cout << ans << std::endl;
};

void testsum() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2 + 1;
        }
    }
    printMatrix(a);
    int ans;
    ans = a.Sum();
    std::cout << ans << std::endl;
};

void testtrace() {
    Matrix<int> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2 + 1;
        }
    }
    printMatrix(a);
    int ans;
    ans = a.trace();
    std::cout << ans << std::endl;
};

void testinv() {
    Matrix<double> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2.0 + 1;
        }
    }
    printMatrix(a);
    Matrix<double> ans(2, 2);
    ans = a.inverse();
    printMatrix(ans);
};

void testeigenv() {
    Matrix<double> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2.0 + 1;
        }
    }
    printMatrix(a);
    std::vector<double> ans;
    ans = a.eigenvalue();
    for (int i = 0; i < ans.size(); i++) {
        std::cout << ans[i] << std::endl;
    }
};

void testdet() {
    Matrix<double> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2.0 + 1;
        }
    }
    printMatrix(a);
    double ans;
    ans = a.determinant();
    std::cout << ans << std::endl;
};

void testres() {
    Matrix<double> a(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.Access(i, j) = i + j * 2.0 + 1;
        }
    }
    printMatrix(a);
    Matrix<double> ans(2, 2);
    ans = a.reshape(1,4);
    printMatrix(ans);
};

void testsli() {
    Matrix<double> a(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a.Access(i, j) = i + j * 2.0 + 1;
        }
    }
    printMatrix(a);
    Matrix<double> ans(2, 2);
    ans = a.slice(0,2,0,2);
    printMatrix(ans);
};

