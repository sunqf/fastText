#include "pairtext.h"
using namespace fasttext;
extern "C" {
PairText* init(char* path) {
  PairText* pairText = new PairText();
  pairText->loadModel(path);
  return pairText;
}

real predictProb(PairText* model, char* first, char* second) {
  if (model == 0) {
    return -1.0;
  }

  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  model->predictProbability(first_str, second_str);
}

bool destroy(PairText* model) {
  if (model != 0) {
    delete model;
  }
}
};
