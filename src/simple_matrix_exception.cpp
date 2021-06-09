//
// Created by GuoYubin on 2021/6/8.
//

#include "simple_matrix_exception.h"

simple_matrix::BadSizeException::BadSizeException(const std::string &arg) : length_error(arg) {}

simple_matrix::BadAccessException::BadAccessException(const std::string &arg) : out_of_range(arg) {}

simple_matrix::ArgumentNotMatchException::ArgumentNotMatchException(const std::string &arg) : invalid_argument(arg) {}
