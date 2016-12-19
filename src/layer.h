//
// Created by sunqf on 2016/12/17.
//

#ifndef FASTTEXT_LAYER_H
#define FASTTEXT_LAYER_H

#include <vector>
#include "matrix.h"
#include "vector.h"

namespace fasttext{
class AverageLayer {
private:
  std::shared_ptr<Matrix> embedding_;
public:
  AverageLayer(std::shared_ptr<Matrix> embedding);
  void compute(const std::vector<int32_t>& input, Vector& output) const;
  void update(const std::vector<int32_t>& input, const Vector& outputGrad);
};

class DropoutLayer {
public:
  void compute(const Vector& input, Vector& output) const;
  void update(const Vector& input, const Vector& outputGrad, Vector& inputGrad);
};

class TransformLayer {
private:
  std::shared_ptr<Matrix> matrix_;
public:
  TransformLayer(std::shared_ptr<Matrix> matrix);
  void compute(const Vector& input, Vector& output) const;
  void update(const Vector& input, const Vector& outputGrad, Vector& inputGrad);
};

/**
 * sigmoid(dot(first, second))
 */
class Similarity {
public:
  real compute(const Vector& first, const Vector& second) const;
  real update(const Vector& first, const Vector& second,
              bool label, real weight,
              real lr,
              Vector& firstGrad, Vector& secondGrad);
};

class Interplate {
public:
  std::shared_ptr<Matrix> matrix_;
public:
  Interplate(std::shared_ptr<Matrix> matrix);
  real compute(const Vector& first, const Vector& second) const;
  real update(const Vector& first, const Vector& second,
              bool label, real weight,
              real lr,
              Vector& firstGrad, Vector& secondGrad);
};
}
#endif //FASTTEXT_LAYER_H
