#include "pairtext.h"
#include "alstext.h"
using namespace fasttext;

extern "C" {
PairText* init(char* path) {
  PairText* pairText = new PairText();
  pairText->loadModel(path);
  return pairText;
}

real predictProb(PairText* model, char* first, char* second) {
  if (model == 0) {
    printf("model hasn't been loaded.\n");
    exit(EXIT_FAILURE);
  }

  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  return model->predictProbability(first_str, second_str);
}

real firstSimilarity(PairText* model, char* first, char* second) {
  if (model == 0) {
    printf("model hasn't been loaded.\n");
    exit(EXIT_FAILURE);
  }
  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  return model->firstSimilarity(first_str, second_str);
}

real secondSimilarity(PairText* model, char* first, char* second) {
  if (model == 0) {
    printf("model hasn't been loaded.\n");
    exit(EXIT_FAILURE);
  }
  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  return model->secondSimilarity(first_str, second_str);
}


void destroy(PairText* model) {
  if (model != 0) {
    delete model;
  }
}
};
