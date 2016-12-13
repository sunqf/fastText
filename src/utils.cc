/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <iostream>
#include <cmath>
#include <ios>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }

  void split(std::string line, char delim, std::vector<std::string>& items) {
    unsigned long start = 0;
    unsigned long end = 0;
    while (start < line.length()) {
      while (end < line.length() && line[end] != delim) ++end;
      items.push_back(line.substr(start, end - start));
      while (end < line.length() && line[end] == delim) ++end;
      start = end;
    }
  }
}

}
