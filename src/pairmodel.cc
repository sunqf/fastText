//
// Created by sunqf on 2016/12/12.
//

#include <iostream>
#include "pairmodel.h"
#include <assert.h>
#include "matrix.h"
#include "args.h"
#include "real.h"

namespace fasttext {

  PairModel::PairModel(std::shared_ptr<Matrix> first_embedding, std::shared_ptr<Matrix> first_w1,
                       std::shared_ptr<Matrix> second_embedding, std::shared_ptr<Matrix> second_w1,
                       std::shared_ptr<Args> args, int32_t seed)
      : first_dropout_input_(2048),
        //first_hidden1_(args->dim),
        first_hidden1_intput_(args->dim),
        first_hidden1_output_(args->dim),
        first_hidden1_grad_(args->dim),
        first_output_(args->dim), first_output_grad_(args->dim),
        second_dropout_input_(2048),
        //second_hidden1_(args->dim),
        second_hidden1_input_(args->dim),
        second_hidden1_output_(args->dim),
        second_hidden1_grad_(args->dim),
        second_output_(args->dim), second_output_grad_(args->dim),
        uniform(0, 1), rng(seed) {
    first_embedding_ = first_embedding;
    first_w1_ = first_w1;
    second_embedding_ = second_embedding;
    second_w1_ = second_w1;
    args_ = args;
    nexamples_ = 1;
    hsz_ = args_->dim;
    isz_ = args_->dim;
    osz_ = args_->dim;
    objLoss_ = 0.0;
    l2Loss_ = 0.0;
    initSigmoid();
    initLog();
  }

/*
  void PairModel::computeHidden(const std::shared_ptr<Matrix> embedding,
                                const std::vector<int32_t>& words,
                                Vector& hidden) const {
    //assert(hidden.size() == hsz_);
    hidden.zero();
    for (auto it = words.cbegin(); it != words.cend(); ++it) {
      hidden.addRow(*embedding, *it);
    }
    hidden.mul(1.0 / words.size());
  }
*/

  void PairModel::computeHidden(const std::shared_ptr<Matrix> embedding,
                                const std::vector<int32_t> &words,
                                Vector &hidden_input,
                                Vector &hidden_output) const {
    hidden_input.zero();
    for (auto it = words.cbegin(); it != words.cend(); ++it) {
      hidden_input.addRow(*embedding, *it);
    }
    hidden_output.l2Norm(hidden_input);
  }

  void PairModel::updateHidden(std::shared_ptr<Matrix> embedding,
                               const std::vector<int32_t>& input,
                               const Vector& hidden_input,
                               Vector& hidden_grad) {

    hidden_grad.l2NormUpdate(hidden_input);
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
      embedding->addRow(hidden_grad, *it, 1.0);
    }
  }

  real PairModel::predict(const std::vector<int32_t>& first,
                          const std::vector<int32_t>& second) const {
    if (first.size() < 30 || second.size() < 30) return 0.0;
    Vector first_hidden1_input(args_->dim), first_hidden1_output(args_->dim);
    Vector second_hidden1_input(args_->dim), second_hidden1_output(args_->dim);
    computeHidden(first_embedding_, first, first_hidden1_input, first_hidden1_output);
    computeHidden(second_embedding_, second, second_hidden1_input, second_hidden1_output);

    Vector first_output(args_->dim), second_output(args_->dim);

    first_output.mul(*first_w1_, first_hidden1_output);
    second_output.mul(*second_w1_, second_hidden1_output);

    return sigmoid(dot(first_output, second_output));
  }

/*
  real PairModel::predict(const std::vector<int32_t>& first,
                          const std::vector<int32_t>& second) {
    computeHidden(first_embedding_, first, first_hidden1_);
    first_output_.mul(*first_w1_, first_hidden1_);

    computeHidden(second_embedding_, second, second_hidden1_);
    second_output_.mul(*second_w1_, second_hidden1_);

    real dot12 = dot(first_output_, second_output_);
    real dot11 = dot(first_output_, first_output_);
    real dot22 = dot(second_output_, second_output_);
    real cosine = dot12 / sqrt(dot11 * dot22);
    return cosine;
  }
*/

/*
  void PairModel::update(const std::vector<int32_t>& input,
                         std::shared_ptr<Matrix> embedding,
                         const Vector& hidden1,
                         Vector& hidden1_grad,
                         std::shared_ptr<Matrix> w1,
                         const Vector& output,
                         const Vector& output_grad) {
    hidden1_grad.zero();
    hidden1_grad.mul(output_grad, *w1);
    w1->addMatrix(output_grad, hidden1);
    w1->addMatrix(*w1, args_->l2);

    hidden1_grad.mul(1.0 / input.size());
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
      embedding->addRow(hidden1_grad, *it, 1.0);
    }
  }
*/


  void PairModel::update(const std::vector<int32_t>& input,
                         std::shared_ptr<Matrix> embedding,
                         const Vector& hidden1_input,
                         const Vector& hidden1_output,
                         Vector& hidden1_grad,
                         std::shared_ptr<Matrix> w1,
                         const Vector& output,
                         const Vector& output_grad) {
    hidden1_grad.zero();
    hidden1_grad.mul(output_grad, *w1);
    w1->addMatrix(output_grad, hidden1_output);
    //w1->addMatrix(*w1, -2 * args_->l2);

    updateHidden(embedding, input, hidden1_input, hidden1_grad);
  }

  void PairModel::update(const std::vector<int32_t>& first_input,
                         const std::vector<int32_t>& second_input,
                         const bool label,
                         real lr,
                         real weight) {
    first_dropout_input_.resize(0);
    for (auto i = 0; i < first_input.size(); i++) {
      if (uniform(rng) > args_->dropout) {
        first_dropout_input_.push_back(first_input[i]);
      }
    }

    second_dropout_input_.resize(0);
    for (auto i = 0; i < second_input.size(); i++) {
      if (uniform(rng) > args_->dropout) {
        second_dropout_input_.push_back(second_input[i]);
      }
    }

    if (first_dropout_input_.size() < 5 || second_dropout_input_.size() < 5) return;

    computeHidden(first_embedding_, first_dropout_input_, first_hidden1_intput_, first_hidden1_output_);
    computeHidden(second_embedding_, second_dropout_input_, second_hidden1_input_, second_hidden1_output_);
    first_output_.mul(*first_w1_, first_hidden1_output_);
    second_output_.mul(*second_w1_, second_hidden1_output_);

    real prob = sigmoid(dot(first_output_, second_output_));

    if (label) {
      objLoss_ += -log(prob) * weight;
    } else {
      objLoss_ += -log(1.0 - prob) * weight;
    }

    //l2Loss_ += 0.5 * args_->l2 * (dot(*first_w1_, *first_w1_) + dot(*second_w1_, *second_w1_)) * weight;
    nexamples_ += 1;

    real alpha = weight * lr * (real(label) - prob);

    // update first
    first_output_grad_.zero();
    first_output_grad_.addVec(second_output_, alpha);

    update(first_dropout_input_,
           first_embedding_, first_hidden1_intput_, first_hidden1_output_, first_hidden1_grad_,
           first_w1_, first_output_, first_output_grad_);


    // update second
    second_output_grad_.zero();
    second_output_grad_.addVec(first_output_, alpha);

    update(second_dropout_input_,
           second_embedding_, second_hidden1_input_, second_hidden1_output_, second_hidden1_grad_,
           second_w1_, second_output_, second_output_grad_);

  }

  void PairModel::getFirstOutput(const std::vector<int32_t>& words, Vector& output) const {
    Vector hidden_input(args_->dim), hidden_output(args_->dim);
    computeHidden(first_embedding_, words, hidden_input, hidden_output);
    output.mul(*first_w1_, hidden_output);
  }

  void PairModel::getSecondOutput(const std::vector<int32_t>& words, Vector& output) const {
    Vector hidden_input(args_->dim), hidden_output(args_->dim);
    computeHidden(second_embedding_, words, hidden_input, hidden_output);
    output.mul(*first_w1_, hidden_output);
  }

  real PairModel::similarity(Vector &first, Vector &second) const {
    real dot12 = dot(first, second);
    real dot11 = dot(first, first);
    real dot22 = dot(second, second);
    real ll = sqrtf(dot11 * dot22);
    if (ll > 0.0) return dot12 / ll;
    return 0.0;
  }

  real PairModel::firstSimilarity(const std::vector<int32_t> &first_words,
                                  const std::vector<int32_t> &second_words) const {
    Vector first_output(args_->dim), second_output(args_->dim);
    getFirstOutput(first_words, first_output);
    getFirstOutput(second_words, second_output);
    return similarity(first_output, second_output);
  }

  real PairModel::secondSimilarity(const std::vector<int32_t> &first_words,
                                   const std::vector<int32_t> &second_words) const {
    Vector first_output(args_->dim), second_output(args_->dim);
    getSecondOutput(first_words, first_output);
    getSecondOutput(second_words, second_output);
    return similarity(first_output, second_output);
  }


  real PairModel::loss(bool label, real prob, real weight) const {
    if (label) {
      return -log(prob) * weight;
    } else {
      return -log(1.0 - prob) * weight;
    }
  }

  void PairModel::initSigmoid() {
    t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
      real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
      t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
    }
  }

  void PairModel::initLog() {
    t_log = new real[LOG_TABLE_SIZE + 1];
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
      real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
      t_log[i] = std::log(x);
    }
  }

  real PairModel::log(real x) const {
    if (x > 1.0) {
      return 0.;
    }
    int i = int(x * LOG_TABLE_SIZE);
    return t_log[i];
  }




  real PairModel::sigmoid(real x) const {
    // x != x  check x is NaN
    if (x != x) {
      return 0.0;
    } else if (x < -MAX_SIGMOID) {
      return 0.0;
    } else if (x > MAX_SIGMOID) {
      return 1.0;
    } else {
      int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
      return t_sigmoid[i];
    }
  }

}
