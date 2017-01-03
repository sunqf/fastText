/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_UTILS_H
#define FASTTEXT_UTILS_H

#include <fstream>
#include <vector>
#include <string>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);

  std::vector<std::string> split(const std::string& line, char delim);

  std::string replace(const std::string &line, char old_char, char new_char);
  class Maths {
  private:
    void initSigmoid();
    void initLog();
  public:
    Maths();

  };
}

}

#endif
