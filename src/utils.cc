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
    int64_t curr = ifs.tellg();
    ifs.seekg(std::streamoff(0), std::ios::end);
    int64_t size = ifs.tellg();
    ifs.seekg(std::streamoff(curr));
    return size;
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }

  std::vector<std::string> split(const std::string& line, char delim) {
    unsigned long start = 0;
    unsigned long end = 0;
    std::vector<std::string> items;
    while (start < line.length()) {
      while (end < line.length() && line[end] != delim) ++end;
      items.push_back(line.substr(start, end - start));
      while (end < line.length() && line[end] == delim) ++end;
      start = end;
    }
    return items;
  }

  std::string replace(const std::string& line, char old_char, char new_char) {
    std::string new_str = line;
    for (auto i = 0; i < new_str.length(); i++) {
      if (new_str[i] == old_char) {
        new_str[i] = new_char;
      }
    }
    return new_str;
  }
}

}
