//
// Created by sunqf on 2017/2/16.
//

#ifndef FASTTEXT_DOCSIM_H
#define FASTTEXT_DOCSIM_H

#include "fasttext.h"

namespace fasttext {
class DocSim {
private:
  std::shared_ptr<Args> args_;
  FastText fastText_;

private:
  void loadModel(const std::string &);
public:
  DocSim(std::shared_ptr<Args> args);

  real predictProbability(const std::string &first, const std::string &second) const;

  real firstSimilarity(const std::string &, const std::string &) const;

  real secondSimilarity(const std::string &, const std::string &) const;

};
}
#endif //FASTTEXT_DOCSIM_H
