/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include <assert.h>

#include <iomanip>

#include <math.h>

#include "matrix.h"
#include "utils.h"

namespace fasttext {

Vector::Vector(int64_t m) {
  m_ = m;
  data_ = new real[m];
}

Vector::~Vector() {
  delete[] data_;
}

int64_t Vector::size() const {
  return m_;
}

void Vector::zero() {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
  }
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] *= a;
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real alpha) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += alpha * A.data_[i * A.n_ + j];
  }
}

void Vector::addVec(const Vector& vec, real a) {
  assert(m_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] += a * vec.data_[i];
  }
}
void Vector::mul(const Matrix& A, const Vector& vec, real alpha) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
    for (int64_t j = 0; j < A.n_; j++) {
      data_[i] += alpha * A.data_[i * A.n_ + j] * vec.data_[j];
    }
  }
}



void Vector::mul(const Vector& vec, const Matrix& A, real alpha) {
  assert(vec.m_ == A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < m_; j++) {
    data_[j] = 0.0;
    for (int64_t i = 0; i < vec.m_; i++) {
      data_[j] += alpha * vec[i] * A.data_[i * A.n_ + j];
    }
  }
}

void Vector::mul(const Matrix& A, const Vector& vec, const Vector& dropout, real alpha) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  assert(vec.m_ == dropout.m_);

  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
    for (int64_t j = 0; j < A.n_; j++) {
      data_[i] += alpha * A.data_[i * A.n_ + j] * vec.data_[j] * dropout[j];
    }
  }
}

void Vector::mul(const Vector& first, const Vector& second) {
  assert(m_ == first.m_);
  assert(m_ == second.m_);
  for (auto i = 0; i < m_; i++) {
    data_[i] = first.data_[i] * second.data_[i];
  }
}
int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < m_; i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

real& Vector::operator[](int64_t i) {
  return data_[i];
}

const real& Vector::operator[](int64_t i) const {
  return data_[i];
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.m_; j++) {
    os << v.data_[j] << ' ';
  }
  return os;
}

real dot(const Vector& first, const Vector& second) {
  assert(first.m_ == second.m_);
  real dist = 0.0;
  for (int64_t i = 0; i < first.size(); i++) {
    dist += first[i] * second[i];
  }
  return dist;
}

real xMy(const Vector& x, const Matrix& m, const Vector& y) {
  assert(x.m_ == m.m_);
  assert(m.n_ == y.m_);
  real dist = 0.0;
  for (int64_t i = 0; i < x.m_; i++) {
    for (int64_t j = 0; j < y.m_; j++) {
      dist += x[i] * m.data_[i * m.n_ + j] * y[j];
    }
  }
  return dist;
}
}
