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
      : first_hidden1_(args->dim), first_output_(args->dim),
        second_hidden1_(args->dim), second_output_(args->dim),
        grad_(args->dim), rng(seed) {
    first_embedding_ = first_embedding;
    first_w1_ = first_w1;
    second_embedding_ = second_embedding;
    second_w1_ = second_w1;
    args_ = args;
    nexamples_ = 1;
    loss_ = 0.0;
    initSigmoid();
    initLog();
  }


  void PairModel::computeHidden(std::shared_ptr<Matrix> embedding,
                                std::vector<int32_t> words,
                                Vector &hidden) {
    //assert(hidden.size() == hsz_);
    hidden.zero();
    for (auto it = words.cbegin(); it != words.cend(); ++it) {
      hidden.addRow(*embedding, *it);
    }
    hidden.mul(1.0 / words.size());
  }

  real PairModel::predict(const std::vector<int32_t>& first,
                          const std::vector<int32_t>& second) {
    computeHidden(first_embedding_, first, first_hidden1_);
    //first_output_.mul(*first_w1_, first_hidden1_);

    computeHidden(second_embedding_, second, second_output_);
    //second_output_.mul(*second_w1_, second_hidden1_);

    //return sigmoid(dot(first_output_, second_output_));
    return sigmoid(dot(first_hidden1_, second_hidden1_));
  }

  void PairModel::update(const std::vector<int32_t>& input,
                         std::shared_ptr<Matrix> embedding,
                         const Vector& hidden1,
                         std::shared_ptr<Matrix> w1,
                         Vector &grad) {
    /*
    for (int32_t i = 0; i < osz_; i++) {
      grad.addRow(*w1, i, 1.0);
      w1->addRow(hidden1, i, 1.0);
    }*/
    grad.mul(1.0 / input.size());
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
      embedding->addRow(grad, *it, 1.0);
    }
  }

  void PairModel::update(const std::vector<int32_t>& first,
                         const std::vector<int32_t>& second,
                         const bool label,
                         real lr) {
    computeHidden(first_embedding_, first, first_hidden1_);
    //first_output_.mul(*first_w1_, first_hidden1_);

    computeHidden(second_embedding_, second, second_hidden1_);
    //second_output_.mul(*second_w1_, second_hidden1_);

    //real prob = sigmoid(dot(first_output_, second_output_));
    real prob = sigmoid(dot(first_hidden1_, second_hidden1_));
    if (label) {
      loss_ += -log(prob);
    } else {
      loss_ += -log(1.0 - prob);
    }
    nexamples_ += 1;

    real alpha = lr * (real(label) - prob);

    // update first
    //Vector first_grad = Vector(second_output_.size());
    Vector first_grad = Vector(second_hidden1_.size());
    first_grad.zero();
    //first_grad.addVec(second_output_, alpha);
    first_grad.addVec(second_hidden1_, alpha);
    update(first, first_embedding_, first_hidden1_, first_w1_, first_grad);
    // update second
    //Vector second_grad = Vector(first_output_.size());
    Vector second_grad = Vector(first_hidden1_.size());
    second_grad.zero();
    //second_grad.addVec(first_output_, alpha);
    second_grad.addVec(first_hidden1_, alpha);
    update(second, second_embedding_, second_hidden1_, second_w1_, second_grad);

    if (nexamples_ % 1000 == 0) {
      std::cout << "first_hidden: " << first_hidden1_ << std::endl;
      std::cout << "first_output: " << first_output_ << std::endl;
      std::cout << "second_hidden: " << second_hidden1_ << std::endl;
      std::cout << "second_output: " << second_output_ << std::endl;
      std::cout << "prob: " << prob << "   label: " << label << " alpha: " << alpha << std::endl;
      std::cout << "first_grad: " << first_grad << std::endl;
      std::cout << "second_grad: " << second_grad << std::endl;
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
    if (x < -MAX_SIGMOID) {
      return 0.0;
    } else if (x > MAX_SIGMOID) {
      return 1.0;
    } else {
      int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
      return t_sigmoid[i];
    }
  }
}
