//
// Created by sunqf on 2016/12/15.
//

#ifndef FASTTEXT_INTERPLATETEXT_H
#define FASTTEXT_INTERPLATETEXT_H

#include <time.h>

#include <atomic>
#include <memory>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "InterplateModel.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

namespace fasttext {
class InterplateText {
private:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> first_dict_;
  std::shared_ptr<Matrix> first_embedding_;

  std::shared_ptr<Dictionary> second_dict_;
  std::shared_ptr<Matrix> second_embedding_;

  std::shared_ptr<Matrix> interplate_;
  std::shared_ptr<InterplateModel> model_;
  std::atomic<int64_t> tokenCount;
  clock_t start;

private:
  void getVector(std::shared_ptr<Dictionary>,
                 std::shared_ptr<Matrix>,
                 Vector&,
                 const std::string&);
  void saveVectors(std::shared_ptr<Dictionary>,
                   std::shared_ptr<Matrix>,
                   std::ofstream&);
public:
  void getFirstVector(Vector&, const std::string&);
  void getSecondVector(Vector&, const std::string&);
  void saveVectors();
  void saveModel();
  void loadModel(const std::string&);
  void loadModel(std::istream&);
  void printInfo(real, real);

  void supervised(InterplateModel&, real,
                  const std::vector<int32_t>&,
                  const std::vector<int32_t>&,
                  const bool label);

  void test(std::istream&);
  void predict(std::istream&);
  real predictProbability(std::istream&) const;
  void wordFirstVectors();
  void textFirstVectors();
  void wordSecondVectors();
  void textSecondVectors();
  void printVectors();
  void trainThread(int32_t);
  void train(std::shared_ptr<Args>);

  void loadVectors(std::string, std::shared_ptr<Dictionary>, std::shared_ptr<Matrix>);
};
}
#endif //FASTTEXT_INTERPLATETEXT_H
