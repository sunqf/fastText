#include "pairtext.h"
#include "alstext.h"
#include "docsim.h"

using namespace fasttext;

extern "C" {
DocSim* init(char* path) {
  DocSim* pairText = new DocSim();
  pairText->loadModel(path);
  return pairText;
}

real predictProb(DocSim* model, char* first, char* second) {
  if (model == 0) {
    printf("model hasn't been loaded.\n");
    exit(EXIT_FAILURE);
  }

  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  return model->predictProbability(first_str, second_str);
}

real firstSimilarity(DocSim* model, char* first, char* second) {
  if (model == 0) {
    printf("model hasn't been loaded.\n");
    exit(EXIT_FAILURE);
  }
  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  return model->firstSimilarity(first_str, second_str);
}

real secondSimilarity(DocSim* model, char* first, char* second) {
  if (model == 0) {
    printf("model hasn't been loaded.\n");
    exit(EXIT_FAILURE);
  }
  std::string first_str = std::string(first);
  std::string second_str = std::string(second);
  return model->secondSimilarity(first_str, second_str);
}


void destroy(DocSim* model) {
  if (model != 0) {
    delete model;
  }
}
};
