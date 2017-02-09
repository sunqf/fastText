//
// Created by sunqf on 2017/2/9.
//

#ifndef FASTTEXT_ACTIVATION_H
#define FASTTEXT_ACTIVATION_H

struct Sigmoid {
  void forward(real *input, real *output, int size) {
    for (auto i = 0; i < size; i++) {
      output[i] = 1 / (1 + exp(-input[i]));
    }
  }

  void backward(real *input, real *grad, int size) {
    for (auto i = 0; i < size; i++) {
      grad[i] *= exp
    }
  }
};
#endif //FASTTEXT_ACTIVATION_H
;