//
// Created by sunqf on 2017/2/16.
//

#ifndef FASTTEXT_DOCSIM_H
#define FASTTEXT_DOCSIM_H

#include "fasttext.h"

namespace fasttext {
class DocSim {
private:
  FastText fastText_;

public:
  void loadModel(const std::string &);

  real predictProbability(const std::string &first, const std::string &second) const;

  real firstSimilarity(const std::string &, const std::string &) const;

  real secondSimilarity(const std::string &, const std::string &) const;

};
}
#endif //FASTTEXT_DOCSIM_H
