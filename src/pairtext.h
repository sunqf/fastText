//
// Created by sunqf on 2016/12/8.
//

#ifndef FASTTEXT_PAIRTEXT_H
#define FASTTEXT_PAIRTEXT_H

#include <time.h>

#include <atomic>
#include <memory>
#include <future>
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
  std::atomic<int64_t> numToken;
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
  bool convertLabel(const std::string&, bool&, real&) const;

  void findSimilarityWords(std::shared_ptr<Dictionary>, std::shared_ptr<Matrix>, std::string&, int32_t) const;

public:
  void getFirstVector(Vector&, const std::string&);
  void getSecondVector(Vector&, const std::string&);
  void saveVectors();
  void saveModel();
  void loadModel(const std::string&);
  void loadModel(std::istream&);
  void printInfo(real, real, real, real);

  void supervised(PairModel&, real,
                  const std::vector<std::pair<int32_t, real>>&,
                  const std::vector<std::pair<int32_t, real>>&,
                  const bool label,
                  real weight);

  void test(std::istream&);
  void predict(std::istream&);
  real predictProbability(std::istream&) const;
  real predictProbability(const std::string& first, const std::string& second) const;
  void wordFirstVectors();
  void textFirstVectors(Vector&);
  void wordSecondVectors();
  void textSecondVectors(Vector&);
  void printVectors();
  void printEmbedding();
  void trainThread(int32_t);
  void validFunc(int32_t, std::shared_ptr<real>, std::shared_ptr<int32_t>) const;
  real valid();
  void train(std::shared_ptr<Args>);

  real firstSimilarity(const std::string&, const std::string&) const;
  real secondSimilarity(const std::string&, const std::string&) const;

  void findSimilarityWords() const;

  void loadVectors(std::string, std::shared_ptr<Dictionary>, std::shared_ptr<Matrix>);
};
}
#endif //FASTTEXT_RECOMMEND_H
