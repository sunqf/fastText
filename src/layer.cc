//
// Created by sunqf on 2016/12/17.
//

#include <cmath>
#include "layer.h"

namespace fasttext {

AverageLayer::AverageLayer(std::shared_ptr<Matrix> embedding): embedding_(embedding) {}

void AverageLayer::compute(const std::vector<int32_t>& input, Vector& output) const {
  output.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    output.addRow(*embedding_, *it);
  }
  output.mul(1.0 / input.size());
}

void AverageLayer::update(const std::vector<int32_t>& input, const Vector& outputGrad) {
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    embedding_->addRow(outputGrad, *it, 1.0 / input.size());
  }
}


void DropoutLayer::compute(const Vector &input, Vector &output) const {

}

void DropoutLayer::update(const Vector &input, const Vector &outputGrad, Vector &inputGrad) {

}
TransformLayer::TransformLayer(std::shared_ptr<Matrix> matrix): matrix_(matrix) {}

void TransformLayer::compute(const Vector& input, Vector& output) const {
  output.zero();
  output.mul(*matrix_, input);
}

void TransformLayer::update(const Vector& input, const Vector& outputGrad, Vector& inputGrad) {
  inputGrad.zero();
  inputGrad.mul(outputGrad, *matrix_);
  matrix_->addMatrix(outputGrad, input);
}

real Similarity::compute(const Vector &first, const Vector &second) const {
  return sigmoid(dot(first, second));
}

real Similarity::update(const Vector &first, const Vector &second,
                        bool label, real weight,
                        real lr,
                        Vector &firstGrad, Vector& secondGrad) {
  real loss = 0.0;
  real prob = compute(first, second);
  if (label) {
    loss = -log(prob) * weight;
  } else {
    loss = -log(1.0 - prob) * weight;
  }
  real alpha = weight * lr * (real(label) - prob);
  firstGrad.zero();
  firstGrad.addVec(second, alpha);
  secondGrad.zero();
  secondGrad.addVec(first, alpha);
  return loss;
}


Interplate::Interplate(std::shared_ptr<Matrix> matrix): matrix_(matrix) {}

real Interplate::compute(const Vector &first, const Vector &second) const {
  return sigmoid(xMy(first, *matrix_, second));
}

real Interplate::update(const Vector &first, const Vector &second,
                        bool label, real weight,
                        real lr,
                        Vector &firstGrad, Vector &secondGrad) {
  real loss = 0.0;
  real prob = compute(first, second);
  if (label) {
    loss = -log(prob) * weight;
  } else {
    loss = -log(1.0 - prob) * weight;
  }

  real alpha = weight * lr * (real(label) - prob);
  firstGrad.zero();
  firstGrad.mul(*matrix_, second, alpha);

  secondGrad.zero();
  secondGrad.mul(first, *matrix_, alpha);

  matrix_->add(first, second, alpha);
}


real Consine::compute(const Vector &first, const Vector &second) const {
 return dot(first, second) / sqrt(dot(first, first) * dot(second, second));
}
real Consine::update(const Vector &first, const Vector &second,
                     bool label, real weight, real lr,
                     Vector &firstGrad, Vector &secondGrad) {
  real square1 = dot(first, first);
  real square2 = dot(second, second);


  // x2 / (len(x1) * len(x2)) - x1 / (square(x1) * square(x2) * len(x1))
  firstGrad.zero();
  firstGrad.addVec(second, 1 / sqrt(square1 * square2));
  firstGrad.addVec(first, 1 / (square1 * square2 * sqrt(square1)));

  // x1 / (len(x1) * len(x2)) - x2 / (square(x1) * square(x2) * len(x2))
  secondGrad.zero();
  secondGrad.addVec(first, 1 / sqrt(square1 * square2));
  secondGrad.addVec(second, 1 / (square1 * square2 * sqrt(square2)));

}
}

