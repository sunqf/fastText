//
// Created by sunqf on 2017/2/16.
//
#include "docsim.h"

#include <iostream>

namespace fasttext {

void DocSim::loadModel(const std::string &path) {
  std::cout << path << std::endl;
  fastText_.loadModel(path);
}


real DocSim::predictProbability(const std::string &first, const std::string &second) const {
  Vector firstVector(512);
  Vector secondVector(512);

  fastText_.getVector(firstVector, first);
  fastText_.getVector(secondVector, second);

  std::cout << firstVector << std::endl;
  std::cout << secondVector << std::endl;

  return cosine(firstVector, secondVector);
}

real DocSim::firstSimilarity(const std::string &first, const std::string &second) const {
  return predictProbability(first, second);
}

real DocSim::secondSimilarity(const std::string &first, const std::string &second) const {
  return predictProbability(first, second);
}
}



