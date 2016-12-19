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
#define MAX_INPUT_SIZE 2048

// warning: non thread-safe
namespace fasttext {

  /**
   *
   *     first     second
   *       |         |
   *     average   average
   *       |         |
   *     output    output
   *       |         |
   *       --sigmoid--
   *
   *
   */

  class PairModel {
  private:
    std::shared_ptr<Matrix> first_embedding_;
    std::shared_ptr<Matrix> first_w1_;
    std::shared_ptr<Matrix> second_embedding_;
    std::shared_ptr<Matrix> second_w1_;

    std::vector<int32_t> first_dropout_input_;
    Vector first_hidden1_;
    Vector first_hidden1_grad_;
    Vector first_dropout1_;
    Vector first_dropout1_output_;
    Vector first_dropout1_grad_;
    Vector first_output_;
    Vector first_output_grad_;

    std::vector<int32_t > second_dropout_input_;
    Vector second_hidden1_;
    Vector second_hidden1_grad_;

    Vector second_dropout1_;
    Vector second_dropout1_output_;
    Vector second_dropout1_grad_;
    Vector second_output_;
    Vector second_output_grad_;
    std::shared_ptr<Args> args_;

    std::uniform_real_distribution<> uniform;

    int32_t isz_;
    int32_t hsz_;
    int32_t osz_;


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
                Vector& hidden1_grad,
                const Vector& dropout1,
                const Vector& dropout1_output,
                Vector& dropout1_grad,
                std::shared_ptr<Matrix> w1,
                const Vector& output,
                const Vector& output_grad);

    void update(const std::vector<int32_t>& input,
                std::shared_ptr<Matrix> embedding,
                const Vector& hidden1,
                Vector& hidden1_grad,
                std::shared_ptr<Matrix> w1,
                const Vector& output,
                const Vector& output_grad);
    void update(const std::vector<int32_t>& first_input,
                const std::vector<int32_t>& second_input,
                const bool label,
                real lr,
                real weight = 1.0);

    real getLoss() { return loss_ / nexamples_; }

    std::minstd_rand rng;
  };
}
#endif //FASTTEXT_PAIRMODEL_H
