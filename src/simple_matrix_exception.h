//
// Created by GuoYubin on 2021/6/8.
//

#ifndef CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_EXCEPTION_H
#define CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_EXCEPTION_H

#include <stdexcept>

namespace simple_matrix {
    class BadSizeException : public std::length_error {
    public:
        BadSizeException(const std::string &arg);
    };
}




#endif //CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_EXCEPTION_H
