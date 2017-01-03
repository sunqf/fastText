//
// Created by sunqf on 2016/12/8.
//

#ifndef FASTTEXT_PAIRTEXT_H
#define FASTTEXT_PAIRTEXT_H

#include <time.h>

#include <atomic>
#include <memory>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "pairmodel.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 512

namespace fasttext {
class PairText {
private:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> first_dict_;
  std::shared_ptr<Matrix> first_embedding_;
  std::shared_ptr<Matrix> first_w1_;

  std::shared_ptr<Dictionary> second_dict_;
  std::shared_ptr<Matrix> second_embedding_;
  std::shared_ptr<Matrix> second_w1_;

  std::shared_ptr<PairModel> model_;
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

  /**
   *
   * @return 如果转换成功返回true, 否则返回false
   */
  bool convertLabel(const std::string&, bool&, real&);
public:
  void getFirstVector(Vector&, const std::string&);
  void getSecondVector(Vector&, const std::string&);
  void saveVectors();
  void saveModel();
  void loadModel(const std::string&);
  void loadModel(std::istream&);
  void printInfo(real, real);

  void supervised(PairModel&, real,
                  const std::vector<int32_t>&,
                  const std::vector<int32_t>&,
                  const bool label,
                  real weight);

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
#endif //FASTTEXT_RECOMMEND_H
