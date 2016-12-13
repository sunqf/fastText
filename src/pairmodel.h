//
// Created by sunqf on 2016/12/12.
//

#ifndef FASTTEXT_PAIRMODEL_H
#define FASTTEXT_PAIRMODEL_H


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
  class PairModel {
  private:
    std::shared_ptr<Matrix> first_embedding_;
    std::shared_ptr<Matrix> first_w1_;
    std::shared_ptr<Matrix> second_embedding_;
    std::shared_ptr<Matrix> second_w1_;

    Vector first_hidden1_;
    Vector second_hidden1_;
    Vector first_output_;
    Vector second_output_;
    std::shared_ptr<Args> args_;

    int32_t hsz_;
    int32_t isz_;
    int32_t osz_;
    Vector grad_;

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
    PairModel(std::shared_ptr<Matrix> first_embedding,
              std::shared_ptr<Matrix> first_w1,
              std::shared_ptr<Matrix> second_embedding,
              std::shared_ptr<Matrix> second_w1,
              std::shared_ptr<Args> args,
              int32_t seed);

    real predict(const std::vector<int32_t>& first,
                 const std::vector<int32_t>& second);

    void update(const std::vector<int32_t>& input,
                           std::shared_ptr<Matrix> embedding,
                           const Vector& hidden1,
                           std::shared_ptr<Matrix> w1,
                           Vector &grad);

    void update(const std::vector<int32_t>& first,
                const std::vector<int32_t>& second,
                const bool label,
                real lr);

    real getLoss() { return loss_; }

    std::minstd_rand rng;
  };
}
#endif //FASTTEXT_PAIRMODEL_H
