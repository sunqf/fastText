//
// Created by sunqf on 2016/12/15.
//

#ifndef FASTTEXT_INTERPLATEMODEL_H
#define FASTTEXT_INTERPLATEMODEL_H

#include <vector>
#include <random>
#include <utility>
#include <memory>

#include "args.h"
#include "matrix.h"
#include "vector.h"
#include "real.h"


#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 512


namespace fasttext {
class InterplateModel {
private:
  std::shared_ptr<Matrix> first_embedding_;
  std::shared_ptr<Matrix> second_embedding_;
  std::shared_ptr<Matrix> interplate_;

  Vector first_hidden1_;
  Vector first_hidden1_grad_;
  Vector second_hidden1_;
  Vector second_hidden1_grad_;
  std::shared_ptr<Args> args_;

  long nexamples_;
  real loss_;

  real* t_sigmoid;
  real* t_log;

private:
  void initSigmoid();
  void initLog();

  real sigmoid(real) const;
  real log(real) const;

  void computeHidden(std::shared_ptr<Matrix> embedding,
                     std::vector<int32_t> words,
                     Vector& hidden);

public:
  InterplateModel(std::shared_ptr<Matrix> first_embedding,
                  std::shared_ptr<Matrix> second_embedding,
                  std::shared_ptr<Matrix> interplate,
                  std::shared_ptr<Args> args_,
                  int32_t seed);

  real predict(const std::vector<int32_t>& first_line,
               const std::vector<int32_t>& second_line);

  void update(const std::vector<int32_t>& first_line,
              const std::vector<int32_t>& second_line,
              const bool label,
              real lr,
              real weight = 1.0);

  real getLoss() { return loss_ / nexamples_; }

  std::minstd_rand rng;
};
}
#endif //FASTTEXT_INTERPLATEMODEL_H
