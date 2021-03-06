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

    std::vector<std::pair<int32_t, real>> first_dropout_input_;
    //Vector first_hidden1_;
    Vector first_hidden1_intput_;
    Vector first_hidden1_output_;
    Vector first_hidden1_grad_;
    Vector first_output_;
    Vector first_output_grad_;

    std::vector<std::pair<int32_t, real>> second_dropout_input_;
    //Vector second_hidden1_;
    Vector second_hidden1_input_;
    Vector second_hidden1_output_;
    Vector second_hidden1_grad_;
    Vector second_output_;
    Vector second_output_grad_;
    std::shared_ptr<Args> args_;

    std::uniform_real_distribution<> uniform;

    int32_t isz_;
    int32_t hsz_;
    int32_t osz_;


    long nexamples_;
    real objLoss_;
    real l2Loss_;

    real* t_sigmoid;
    real* t_log;

  private:
    void initSigmoid();
    void initLog();

    real sigmoid(real) const;
    real log(real) const;

    void computeHidden(const std::shared_ptr<Matrix> embedding,
                       const std::vector<std::pair<int32_t,real>>& words,
                       Vector& hidden_input,
                       Vector& hidden_output) const;

    void updateHidden(std::shared_ptr<Matrix> embedding,
                      const std::vector<std::pair<int32_t, real>>& input,
                      const Vector& hidden_input,
                      Vector& hidden1_grad);

    void getFirstOutput(const std::vector<std::pair<int32_t, real>>& words, Vector& output) const;
    void getSecondOutput(const std::vector<std::pair<int32_t, real>>& words, Vector& output) const;

    real similarity(Vector& first, Vector& second) const;
  public:
    PairModel(std::shared_ptr<Matrix> first_embedding,
              std::shared_ptr<Matrix> first_w1,
              std::shared_ptr<Matrix> second_embedding,
              std::shared_ptr<Matrix> second_w1,
              std::shared_ptr<Args> args,
              int32_t seed);

    real predict(const std::vector<std::pair<int32_t, real>>& first,
                 const std::vector<std::pair<int32_t, real>>& second) const;

    void update(const std::vector<std::pair<int32_t, real>>& input,
                std::shared_ptr<Matrix> embedding,
                const Vector& hidden1_input,
                const Vector& hidden1_output,
                Vector& hidden1_grad,
                std::shared_ptr<Matrix> w1,
                const Vector& output,
                const Vector& output_grad);

    void update(const std::vector<std::pair<int32_t, real>>& first_input,
                const std::vector<std::pair<int32_t, real>>& second_input,
                const bool label,
                real lr,
                real weight = 1.0);


    real firstSimilarity(const std::vector<std::pair<int32_t, real>>& first_words,
                         const std::vector<std::pair<int32_t, real>>& second_words) const;

    real secondSimilarity(const std::vector<std::pair<int32_t, real>>& first_words,
                          const std::vector<std::pair<int32_t, real>>& second_words) const;

    real getObjLoss() { return objLoss_ / nexamples_; }
    real getL2Loss() { return l2Loss_ / nexamples_; }

    real getLoss() { return (objLoss_ + l2Loss_) / nexamples_; }

    real loss(bool label, real prob, real weight) const;

    std::minstd_rand rng;
  };
}
#endif //FASTTEXT_PAIRMODEL_H
