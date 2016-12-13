//
// Created by sunqf on 2016/12/12.
//
#include "pairtext.h"

#include <fenv.h>
#include <math.h>

#include <istream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>

#include "utils.h"

namespace fasttext {

  void PairText::getVector(std::shared_ptr<Dictionary> dict,
                           std::shared_ptr<Matrix> input,
                           Vector& vec,
                           const std::string& word) {
    const std::vector<int32_t>& ngrams = dict->getNgrams(word);
    vec.zero();
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
      vec.addRow(*first_embedding_, *it);
    }
    if (ngrams.size() > 0) {
      vec.mul(1.0 / ngrams.size());
    }
  }

  void PairText::getFirstVector(Vector& vec, const std::string& word) {
    getVector(first_dict_, first_embedding_, vec, word);
  }

  void PairText::getSecondVector(Vector& vec, const std::string& word) {
    getVector(second_dict_, second_embedding_, vec, word);
  }

  void PairText::saveVectors(std::shared_ptr<Dictionary> dict,
                             std::shared_ptr<Matrix> embedding,
                             std::ofstream& ofs) {
    ofs << dict->nwords() << " " << args_->dim << std::endl;
    Vector vec(args_->dim);
    for (int32_t i = 0; i < dict->nwords(); i++) {
      std::string word = dict->getWord(i);
      getVector(dict, embedding, vec, word);
      ofs << word << " " << vec << std::endl;
    }
  }
  void PairText::saveVectors() {
    std::ofstream ofs(args_->output + ".vec");
    if (!ofs.is_open()) {
      std::cout << "Error opening file for saving vectors." << std::endl;
      exit(EXIT_FAILURE);
    }

    saveVectors(first_dict_, first_embedding_, ofs);

    ofs << "-------------------------------" << std::endl;

    saveVectors(second_dict_, second_embedding_, ofs);

    ofs.close();
  }

  void PairText::saveModel() {
    std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
    if (!ofs.is_open()) {
      std::cerr << "Model file cannot be opened for saving!" << std::endl;
      exit(EXIT_FAILURE);
    }
    args_->save(ofs);
    first_dict_->save(ofs);
    first_embedding_->save(ofs);
    first_w1_->save(ofs);
    second_dict_->save(ofs);
    second_embedding_->save(ofs);
    second_w1_->save(ofs);
    ofs.close();
  }

  void PairText::loadModel(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs.is_open()) {
      std::cerr << "Model file cannot be opened for loading!" << std::endl;
      exit(EXIT_FAILURE);
    }
    loadModel(ifs);
    ifs.close();
  }

  void PairText::loadModel(std::istream& in) {
    args_ = std::make_shared<Args>();
    first_dict_ = std::make_shared<Dictionary>(args_);
    first_embedding_ = std::make_shared<Matrix>();
    first_w1_ = std::make_shared<Matrix>();
    second_dict_ = std::make_shared<Dictionary>(args_);
    second_embedding_ = std::make_shared<Matrix>();
    second_w1_ = std::make_shared<Matrix>();

    args_->load(in);
    first_dict_->load(in);
    first_embedding_->load(in);
    first_w1_->load(in);
    second_dict_->load(in);
    second_embedding_->load(in);
    second_w1_->load(in);

    model_ = std::make_shared<PairModel>(first_embedding_, first_w1_, second_embedding_, second_w1_, args_, 0);

  }

  void PairText::printInfo(real progress, real loss) {
    real t = real(clock() - start) / CLOCKS_PER_SEC;
    real wst = real(tokenCount) / t;
    real lr = args_->lr * (1.0 - progress);
    int eta = int(t / progress * (1 - progress) / args_->thread);
    int etah = eta / 3600;
    int etam = (eta - etah * 3600) / 60;
    std::cout << std::fixed;
    std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
    std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
    std::cout << "  lr: " << std::setprecision(6) << lr;
    std::cout << "  loss: " << std::setprecision(6) << loss;
    std::cout << "  eta: " << etah << "h" << etam << "m ";
    std::cout << std::flush;
  }

  void PairText::supervised(PairModel& model, real lr,
                            const std::vector<int32_t>& first,
                            const std::vector<int32_t>& second,
                            const bool label) {
    if (first.size() == 0 || second.size() == 0) return;

    model.update(first, second, label, lr);
  }

  void PairText::test(std::istream& in) {
    int32_t nexamples = 0;
    int32_t nTrue = 0; // TT + TF
    int32_t nTT = 0; // TT
    double precision = 0.0;
    int32_t label;
    std::vector<int32_t> first_line;
    std::vector<int32_t> second_line;

    while (in.peek() != EOF) {
      std::string line;
      getline(in, line);
      std::vector<std::string> items;
      utils::split(line, '\t', items);
      in >> label;
      first_dict_->getLine(items[1], first_line, model_->rng);
      first_dict_->addNgrams(first_line, args_->wordNgrams);

      second_dict_->getLine(items[2], second_line, model_->rng);
      second_dict_->addNgrams(second_line, args_-> wordNgrams);
      if (first_line.size() > 0 && second_line.size() > 0) {
        real prob = model_->predict(first_line, second_line);
        if (label == 1) {
          nTrue++;
          if (prob > 0.5) {
            precision += 1.0;
            nTT++;
          }
        } else if (prob < 0.5 && label == 0) precision += 1.0;
        nexamples++;
      }
    }
    std::cout << std::setprecision(3);
    std::cout << "Prec" << ": " << precision / nexamples << std::endl;
    std::cout << "Recal" << ": " << nTT / nTrue << std::endl;
    std::cout << "Number of examples: " << nexamples << std::endl;
  }

  real PairText::predictProbability(std::istream& in) const {
    int32_t label;
    std::vector<int32_t> first_words;
    std::vector<int32_t> second_words;

    std::string line;
    getline(in, line);
    std::vector<std::string> items;
    utils::split(line, '\t', items);
    first_dict_->getLine(items[1], first_words, model_->rng);
    first_dict_->addNgrams(first_words, args_->wordNgrams);

    second_dict_->getLine(items[2], second_words, model_->rng);
    second_dict_->addNgrams(second_words, args_->wordNgrams);
    if (first_words.empty() || second_words.empty()) return 0.0;
    return model_->predict(first_words, second_words);
  }

  void PairText::predict(std::istream& in) {
    std::vector<std::pair<real,std::string>> predictions;
    while (in.peek() != EOF) {
      real prob = predictProbability(in);
      std::cout << ' ' << prob;
      std::cout << std::endl;
    }
  }

  void PairText::wordFirstVectors() {
    std::string word;
    Vector vec(args_->dim);
    while (std::cin >> word) {
      getFirstVector(vec, word);
      std::cout << word << " " << vec << std::endl;
    }
  }

  void PairText::textFirstVectors() {
    std::vector<int32_t> line, labels;
    Vector vec(args_->dim);
    while (std::cin.peek() != EOF) {
      first_dict_->getLine(std::cin, line, labels, model_->rng);
      first_dict_->addNgrams(line, args_->wordNgrams);
      vec.zero();
      for (auto it = line.cbegin(); it != line.cend(); ++it) {
        vec.addRow(*first_embedding_, *it);
      }
      if (!line.empty()) {
        vec.mul(1.0 / line.size());
      }
      std::cout << vec << std::endl;
    }
  }

  void PairText::wordSecondVectors() {
    std::string word;
    Vector vec(args_->dim);
    while (std::cin >> word) {
      getSecondVector(vec, word);
      std::cout << word << " " << vec << std::endl;
    }
  }

  void PairText::textSecondVectors() {
    std::vector<int32_t> line, labels;
    Vector vec(args_->dim);
    while (std::cin.peek() != EOF) {
      second_dict_->getLine(std::cin, line, labels, model_->rng);
      second_dict_->addNgrams(line, args_->wordNgrams);
      vec.zero();
      for (auto it = line.cbegin(); it != line.cend(); ++it) {
        vec.addRow(*second_embedding_, *it);
      }
      if (!line.empty()) {
        vec.mul(1.0 / line.size());
      }
      std::cout << vec << std::endl;
    }
  }

  void PairText::printVectors() {
    if (args_->model == model_name::sup) {
      textFirstVectors();
      textSecondVectors();
    } else {
      wordFirstVectors();
      wordSecondVectors();
    }
  }


  void PairText::trainThread(int32_t threadId) {
    std::ifstream ifs(args_->input + ".label");
    utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

    std::string line;
    getline(ifs, line);
    std::cout << line << std::endl;
    PairModel model(first_embedding_, first_w1_, second_embedding_, second_w1_, args_, threadId);

    const int64_t ntokens = first_dict_->ntokens() + second_dict_->ntokens();
    int64_t localTokenCount = 0;
    std::vector<int32_t> first_words, second_words;
    bool label;
    while (tokenCount < args_->epoch * ntokens) {
      real progress = real(tokenCount) / (args_->epoch * ntokens);
      real lr = args_->lr * (1.0 - progress);
      getline(ifs, line);
      std::vector<std::string> items;
      utils::split(line, '\t', items);

      if (items[0] == "0") label = false;
      else if(items[0] == "1") label = true;

      localTokenCount += first_dict_->getLine(items[1], first_words, model.rng);
      localTokenCount += second_dict_->getLine(items[2], second_words, model.rng);
      if (args_->model == model_name::sup) {
        first_dict_->addNgrams(first_words, args_->wordNgrams);
        second_dict_->addNgrams(second_words, args_->wordNgrams);
        supervised(model, lr, first_words, second_words, label);
      }
      if (localTokenCount > args_->lrUpdateRate) {
        tokenCount += localTokenCount;
        localTokenCount = 0;
        if (threadId == 0 && args_->verbose > 1) {
          printInfo(progress, model.getLoss());
        }
      }
    }
    if (threadId == 0 && args_->verbose > 0) {
      printInfo(1.0, model.getLoss());
      std::cout << std::endl;
    }
    ifs.close();
  }

  void PairText::loadVectors(std::string filename,
                             std::shared_ptr<Dictionary> dict,
                             std::shared_ptr<Matrix> embedding) {
    std::ifstream in(filename);
    std::vector<std::string> words;
    std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
    int64_t n, dim;
    if (!in.is_open()) {
      std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    in >> n >> dim;
    if (dim != args_->dim) {
      std::cerr << "Dimension of pretrained vectors does not match -dim option"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    mat = std::make_shared<Matrix>(n, dim);
    for (size_t i = 0; i < n; i++) {
      std::string word;
      in >> word;
      words.push_back(word);
      first_dict_->add(word);
      for (size_t j = 0; j < dim; j++) {
        in >> mat->data_[i * dim + j];
      }
    }
    in.close();

    dict->threshold(1, 0);
    embedding = std::make_shared<Matrix>(dict->nwords()+args_->bucket, args_->dim);
    embedding->uniform(1.0 / args_->dim);

    for (size_t i = 0; i < n; i++) {
      int32_t idx = dict->getId(words[i]);
      if (idx < 0 || idx >= dict->nwords()) continue;
      for (size_t j = 0; j < dim; j++) {
        embedding->data_[idx * dim + j] = mat->data_[i * dim + j];
      }
    }
  }

  void PairText::train(std::shared_ptr<Args> args) {
    args_ = args;
    first_dict_ = std::make_shared<Dictionary>(args_);
    second_dict_ = std::make_shared<Dictionary>(args_);
    if (args_->input == "-") {
      // manage expectations
      std::cerr << "Cannot use stdin for training!" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::ifstream first_fs(args_->input + ".first");
    if (!first_fs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    first_dict_->readFromFile(first_fs);
    first_fs.close();

    std::ifstream second_fs(args_->input + ".second");
    if (!second_fs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    second_dict_->readFromFile(second_fs);
    second_fs.close();

    if (args_->pretrainedVectors.size() != 0) {
      loadVectors(args_->pretrainedVectors, first_dict_, first_embedding_);
    } else {
      first_embedding_ = std::make_shared<Matrix>(first_dict_->nwords() + args_->bucket, args_->dim);
      first_embedding_->uniform(1.0 / args_->dim);
    }

    if (args_->pretrainedVectors.size() != 0) {
      loadVectors(args_->pretrainedVectors, second_dict_, second_embedding_);
    } else {
      second_embedding_ = std::make_shared<Matrix>(second_dict_->nwords() + args_->bucket, args_->dim);
      second_embedding_->uniform(1.0 / args_->dim);
    }

    first_w1_ = std::make_shared<Matrix>(args_->dim, args_->dim);
    second_w1_ = std::make_shared<Matrix>(args_->dim, args_->dim);

    first_w1_->uniform(1.0 / args_->dim);
    second_w1_->uniform(1.0 / args_->dim);

    start = clock();
    tokenCount = 0;
    std::vector<std::thread> threads;
    for (int32_t i = 0; i < args_->thread; i++) {
      threads.push_back(std::thread([=]() { trainThread(i); }));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
      it->join();
    }
    model_ = std::make_shared<PairModel>(first_embedding_,
                                         first_w1_,
                                         second_embedding_,
                                         second_w1_,
                                         args_, 0);

    saveModel();
    if (args_->model != model_name::sup) {
      saveVectors();
    }
  }

}
