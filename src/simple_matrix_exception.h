#ifndef CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_EXCEPTION_H
#define CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_EXCEPTION_H

#include <stdexcept>

namespace simple_matrix {
    class BadSizeException : public std::length_error {
    public:
        BadSizeException(const std::string &arg);
    };

    class BadAccessException : public std::out_of_range {
    public:
        BadAccessException(const std::string &arg);
    };

    class ArgumentNotMatchException : public std::invalid_argument {
    public:
        ArgumentNotMatchException(const std::string &arg);
    };

}




#endif //CS205_PROJECT_SIMPLE_MATRIX_SIMPLE_MATRIX_EXCEPTION_H
