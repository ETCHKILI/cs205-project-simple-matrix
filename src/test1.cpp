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
    std::cout << "matrix values\n";
    for (int i = 0; i < mat.getRowSize(); ++i) {
        for (int j = 0; j < mat.getColumnSize(); ++j) {
            std::cout << mat.Access(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void testCVtoM();

void testMtoCV();

void testeigen();


using std::cout;
int main() {
    testadd();
    testsub();
    testmul1();
    testmul2(); // matrix * constant;

    testdot();
    testtrans();
    testconj();
    testMAX();
    testMIN();
    testsum();
    testtrace();
    testeigenv();
    testinv();
    testdet();
    testres();
    testsli();
    testconv();
    testCVtoM();
    testMtoCV();

    testeigen();
    return 0;
}

void testCVtoM() {
    cout << "----------\n";
    cv::Mat a(2, 2, CV_32S);

    cout << "cv Mat\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << (a.at<int>(i, j) = i + j) << ' ';
        }
        std::cout << '\n';
    }
    Matrix b = simple_matrix::CvMatToMatrix(a);
    printMatrix(b) ;
}

void testMtoCV() {
    cout << "----------\n";
    Matrix a(2, 2, 0);

    cout << "matrix values\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << (a[i][j] = i + j) << ' ';
        }
        std::cout << '\n';
    }

    auto b = cv::Mat_<int>(a);
    cout << "cv Mat\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << (a[i][j] = i + j) << ' ';
        }
        std::cout << '\n';
    }
}

void testadd() {
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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
    cout << "----------\n";
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

void testeigen(){
    cout << "----------\n";
    Matrix a(3,3,(double)0);
    a.Access(0,0) = 1;
    a.Access(0,1) = 2;
    a.Access(0,2) = 3;
    a.Access(1,0) = 6;
    a.Access(1,1) = 5;
    a.Access(1,2) = 4;
    a.Access(2,0) = 7;
    a.Access(2,1) = 8;
    a.Access(2,2) = 9;
    auto v = a.eigen();
    for (int i = 0; i < v.size(); ++i) {
        std::cout << "eigenvalue:\n";
        std::cout<<v[i].first<<"\n";
        std::cout << "eigenvector values:\n";
        for (int j = 0; j < v[i].second.size(); ++j) {
            std::cout<<v[i].second[j]<<" ";
        }
        std::cout<<"\n";
    }
}

