//
// Created by sunqf on 2017/2/16.
//
#include "docsim.h"

namespace fasttext {

DocSim::DocSim(std::shared_ptr<Args> args): args_(args) {
  loadModel(args_->input);
}
void DocSim::loadModel(const std::string &path) {
  fastText_.loadModel(path);
}


real DocSim::predictProbability(const std::string &first, const std::string &second) const {
  Vector firstVector(args_->dim);
  Vector secondVector(args_->dim);

  fastText_.getVector(firstVector, first);
  fastText_.getVector(secondVector, second);

  return cosine(firstVector, secondVector);
}

real DocSim::firstSimilarity(const std::string &first, const std::string &second) const {
  return predictProbability(first, second);
}

real DocSim::secondSimilarity(const std::string &first, const std::string &second) const {
  return predictProbability(first, second);
}
}



