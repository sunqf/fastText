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
#include <future>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

#include "utils.h"

namespace fasttext {

  void PairText::getVector(std::shared_ptr<Dictionary> dict,
                           std::shared_ptr<Matrix> embedding,
                           Vector& vec,
                           const std::string& word) {
    const std::vector<int32_t>& ngrams = dict->getNgrams(word);
    vec.zero();
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
      vec.addRow(*embedding, *it);
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

  void PairText::printInfo(real progress, real loss, real objLoss, real l2Loss) {
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
    std::cout << "  loss: " << std::setprecision(6) << loss << "  obj loss: " <<  std::setprecision(6) << objLoss << "  l2 loss:  " << std::setprecision(6) << l2Loss;
    std::cout << "  eta: " << etah << "h" << etam << "m ";
    std::cout << std::flush;
  }

  void PairText::supervised(PairModel& model, real lr,
                            const std::vector<std::pair<int32_t, real>>& first,
                            const std::vector<std::pair<int32_t, real>>& second,
                            const bool label,
                            real weight) {
    if (first.size() == 0 || second.size() == 0) return;

    model.update(first, second, label, lr, weight);
  }

  void PairText::test(std::istream& in) {
    int32_t nexamples = 0;
    int32_t nTrue = 0; // TT + TF
    double nTT = 0; // TT
    double precision = 0.0;
    std::vector<std::pair<int32_t, real>> first_line;
    std::vector<std::pair<int32_t, real>> second_line;

    while (in.peek() != EOF) {
      std::string first, second, step;
      getline(in, first);
      first_dict_->getWords(first, first_line, args_->wordNgrams, model_->rng);
      getline(in, second);
      second_dict_->getWords(second, second_line, args_->wordNgrams, model_->rng);

      getline(in, step);
      bool label;
      real weight = 1.0;
      if (convertLabel(step, label, weight) && first_line.size() > 30 && second_line.size() > 30) {
        real prob = model_->predict(first_line, second_line);
        std::cout << first << std::endl;
        std::cout << second << std::endl;
        std::cout << step << " " << prob << std::endl;
        if (label == true) {
          nTrue++;
          if (prob >= 0.5) {
            precision += 1.0;
            nTT++;
          }
        } else if (prob < 0.5 && label == false) {
          precision += 1.0;
        }
        nexamples++;
      }
    }
    std::cout << std::setprecision(3);
    std::cout << "nexamples" << ": " << nexamples << std::endl;
    std::cout << "nTrue" << ": " << nTrue << std::endl;
    std::cout << "nTT" << ": " << nTT << std::endl;
    std::cout << "Prec" << ": " << precision / nexamples << std::endl;
    std::cout << "Recal" << ": " << nTT / nTrue << std::endl;
    std::cout << "Number of examples: " << nexamples << std::endl;
  }

  real PairText::predictProbability(std::istream& in) const {
    int32_t label;
    std::vector<std::pair<int32_t, real>> first_words;
    std::vector<std::pair<int32_t, real>> second_words;

    std::string first_line, second_line;
    getline(in, first_line);
    getline(in, second_line);
    first_dict_->getWords(first_line, first_words, args_->wordNgrams, model_->rng);
    second_dict_->getWords(second_line, second_words, args_->wordNgrams, model_->rng);
    if (first_words.size() > 30 || second_words.size() > 30) return 0.0;
    return model_->predict(first_words, second_words);
  }

  real PairText::predictProbability(const std::string& first, const std::string& second) const {
    if (first.length() == 0 || second.length() == 0) return 0.0;
    std::vector<std::pair<int32_t, real>> first_words;
    std::vector<std::pair<int32_t, real>> second_words;
    first_dict_->getWords(first, first_words, args_->wordNgrams, model_->rng);
    second_dict_->getWords(second, second_words, args_->wordNgrams, model_->rng);
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

  void PairText::textFirstVectors(Vector& vec) {
    std::vector<int32_t> line, labels;
    if (std::cin.peek() != EOF) {
      first_dict_->getLine(std::cin, line, labels, model_->rng);
      first_dict_->addNgrams(line, args_->wordNgrams);
      vec.zero();
      for (auto it = line.cbegin(); it != line.cend(); ++it) {
        vec.addRow(*first_embedding_, *it);
      }
      if (!line.empty()) {
        vec.mul(1.0 / line.size());
      }
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

  void PairText::textSecondVectors(Vector& vec) {
    std::vector<int32_t> line, labels;
    if (std::cin.peek() != EOF) {
      second_dict_->getLine(std::cin, line, labels, model_->rng);
      second_dict_->addNgrams(line, args_->wordNgrams);
      vec.zero();
      for (auto it = line.cbegin(); it != line.cend(); ++it) {
        vec.addRow(*second_embedding_, *it);
      }
      if (!line.empty()) {
        vec.mul(1.0 / line.size());
      }
    }
  }

  void PairText::printVectors() {
    if (args_->model == model_name::sup) {
      Vector first_hidden(args_->dim), second_hidden(args_->dim);
      Vector first_output(args_->dim), second_output(args_->dim);
      while (std::cin.peek() != EOF) {
        textFirstVectors(first_hidden);
        first_output.mul(*first_w1_, first_hidden);
        textSecondVectors(second_hidden);
        second_output.mul(*second_w1_, second_hidden);
    	std::cout << first_hidden << std::endl << first_output << std::endl;
        std::cout << second_hidden << std::endl << second_output << std::endl;
        std::cout << dot(first_output, second_output) << " " << dot(first_output, first_output) << " " << dot(second_output, second_output) << std::endl;        
      }
    } else {
      wordFirstVectors();
      wordSecondVectors();
    }
  }

  bool PairText::convertLabel(const std::string &text, bool &label, real &weight) const {
    if (text == "INTERVIEW") {
      label = true;
      weight = 0.25;
      return true;
    } else if (text == "REFUSE") {
      label = false;
      weight = 1.0;
      return true;
    } else if (text == "ACCEPT_INTERVIEW") {
      label = true;
      weight = 5.0;
      return true;
    }
    return false;
  }
  void PairText::trainThread(int32_t threadId) {
    std::ifstream ifs(args_->input);
    utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);
    int64_t endPos = std::min(utils::size(ifs), (threadId + 1) * utils::size(ifs) / args_->thread);

    // 去掉第一个不完整的样本
    std::string line;
    while (getline(ifs, line)) {
      if (line == "REFUSE" || line == "INTERVIEW" || line == "ACCEPT_INTERVIEW") break;
    }
    PairModel model(first_embedding_, first_w1_, second_embedding_, second_w1_, args_, threadId);

    int64_t localTokenCount = 0;
    std::string first, second, third;
    std::vector<std::pair<int32_t, real>> first_words, second_words;
    bool label;
    real weight = 1.0;
    while (ifs.tellg() > 0 && ifs.tellg() < endPos) {
      real progress = real(tokenCount) / (args_->epoch * numToken);
      real lr = args_->lr * (1.0 - progress);

      getline(ifs, first);
      getline(ifs, second);
      getline(ifs, third);
      if (first.empty() || second.empty() || third.empty()) continue;

      int64_t tokenCount1 = first_dict_->getWords(first, first_words, args_->wordNgrams, model.rng);

      int64_t tokenCount2 = second_dict_->getWords(second, second_words, args_->wordNgrams, model.rng);
      
      if (!convertLabel(third, label, weight) || tokenCount1 < 30 || tokenCount2 < 30) continue;

      localTokenCount += tokenCount1 + tokenCount2;

      //first_dict_->addNgrams(first_words, args_->wordNgrams);
      //second_dict_->addNgrams(second_words, args_->wordNgrams);
      supervised(model, lr, first_words, second_words, label, weight);

      if (localTokenCount > args_->lrUpdateRate) {
        tokenCount += localTokenCount;
        localTokenCount = 0;
        if (threadId == 0 && args_->verbose > 1) {
          printInfo(progress, model.getLoss(), model.getObjLoss(), model.getL2Loss());
        }
      }
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
      dict->add(word);
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

  real PairText::firstSimilarity(const std::string &first, const std::string &second) const {
    if (first_dict_ == nullptr || second_dict_ == nullptr || model_ == nullptr ) {
      std::cerr << "There are some problems in model" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::vector<std::pair<int32_t, real>> first_words, second_words;
    first_dict_->getWords(first, first_words, args_->wordNgrams, model_->rng);
    first_dict_->getWords(second, second_words, args_->wordNgrams, model_->rng);

    return model_->firstSimilarity(first_words, second_words);
  }

  real PairText::secondSimilarity(const std::string &first, const std::string &second) const {
    if (first_dict_ == nullptr || second_dict_ == nullptr || model_ == nullptr) {
      std::cerr << "There are some problems in model" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::vector<std::pair<int32_t, real>> first_words, second_words;
    second_dict_->getWords(first, first_words, args_->wordNgrams, model_->rng);
    second_dict_->getWords(second, second_words, args_->wordNgrams, model_->rng);

    return model_->secondSimilarity(first_words, second_words);
  }

  bool compare(std::pair<int32_t, real> first,
               std::pair<int32_t, real> second) {
    return (first.second > second.second);
  }

  void PairText::findSimilarityWords(std::shared_ptr<Dictionary> dict,
                                     std::shared_ptr<Matrix> embedding,
                                     std::string& word, int32_t n) const {
    int32_t id = dict->getId(word);
    std::vector<std::pair<int32_t, real>> id2prob(dict->nwords());
    Vector vec(args_->dim);
    embedding->getRow(id, vec);
    Vector curVec(args_->dim);
    for (int32_t i = 0; i < dict->nwords(); i++) {
      if (i != id) {
        embedding->getRow(i, curVec);
        real cos = cosine(vec, curVec);
        id2prob[i] = std::make_pair(i, cos);
      }
    }
    std::sort(id2prob.begin(), id2prob.end(), compare);
    for (int i = 0; i < n; i++) {
      std::cout << dict->getWord(id2prob[i].first) << " " << dict->getCount(id2prob[i].first) << " " << id2prob[i].second << std::endl;
    }
  }

  void PairText::findSimilarityWords() const {
    int32_t dir;
    std::string word;
    int32_t n = 10;
    while (std::cin >> dir >> word >> n) {
      if (dir == 1) {
        findSimilarityWords(first_dict_, first_embedding_, word, n);
      } else if (dir == 2) {
        findSimilarityWords(second_dict_, second_embedding_, word, n);
      }
    }
  }

  void PairText::validFunc(int32_t threadId, std::shared_ptr<real> pLoss, std::shared_ptr<int32_t> nexamples) const {
    std::ifstream ifs(args_->valid);
    utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);
    int64_t endPos = std::min(utils::size(ifs), (threadId + 1) * utils::size(ifs) / args_->thread);
    // 去掉第一个不完整的样本
    std::string line;
    while (getline(ifs, line)) {
      if (line == "REFUSE" || line == "INTERVIEW" || line == "ACCEPT_INTERVIEW") break;
    }
    PairModel model(first_embedding_, first_w1_, second_embedding_, second_w1_, args_, threadId);

    const int64_t ntokens = first_dict_->ntokens() + second_dict_->ntokens();
    real localLoss = 0.0;

    std::string first, second, third;
    std::vector<std::pair<int32_t, real>> first_words, second_words;
    bool label;
    real weight = 1.0;
    while (ifs.tellg() > 0 && ifs.tellg() < endPos) {
      getline(ifs, first);
      getline(ifs, second);
      getline(ifs, third);
      if (first.empty() || second.empty() || third.empty()) continue;
      first_dict_->getWords(first, first_words, args_->wordNgrams, model.rng);

      second_dict_->getWords(second, second_words, args_->wordNgrams, model.rng);

      if (!convertLabel(third, label, weight) || first_words.size() < 30 || second_words.size() < 30) continue;

      real prob = model.predict(first_words, second_words);

      localLoss += model.loss(label, prob, weight);

      (*nexamples)++;
    }
    *pLoss = localLoss;
    ifs.close();
  }

  real PairText::valid() {
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<real>> losses;
    std::vector<std::shared_ptr<int32_t>> numbers;
    real validLoss = 0.0;
    int32_t nexamples = 0;
    threads.clear();
    for (int32_t i = 0; i < args_->thread; i++) {
      std::shared_ptr<real> pLoss = std::shared_ptr<real>(new real(0.0));
      std::shared_ptr<int32_t> pNum = std::shared_ptr<int32_t>(new int32_t(0));
      losses.push_back(pLoss);
      numbers.push_back(pNum);
      //threads.push_back(std::thread(validFunc, i, std::move(lossPromise)));
      threads.push_back(std::thread([=]() { validFunc(i, pLoss, pNum); }));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
      it->join();
    }
    for (int32_t i = 0; i < args_->thread; i++) {
      validLoss += *(losses[i]);
      nexamples += *(numbers[i]);
    }
    return validLoss / nexamples;
  }

  void PairText::train(std::shared_ptr<Args> args) {
    if (args->input == "-") {
      // manage expectations
      std::cerr << "Cannot use stdin for training!" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::ifstream model_fs(args->output, std::ifstream::binary);
    if (model_fs.is_open()) {
      loadModel(model_fs);
      args_ = args;
    } else {
      args_ = args;
      first_dict_ = std::make_shared<Dictionary>(args_);
      second_dict_ = std::make_shared<Dictionary>(args_);
      std::ifstream first_fs(args_->dict + ".first");
      if (!first_fs.is_open()) {
        std::cerr << "first dict " << args_->dict + ".first" << " cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
      }
      first_dict_->build(first_fs);
      first_fs.close();
  
      std::cout << "first dict " << first_dict_->nwords() << std::endl;
  
      std::ifstream second_fs(args_->dict + ".second");
      if (!second_fs.is_open()) {
        std::cerr << "second dict " << args_->dict + ".second" << " cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
      }
      second_dict_->build(second_fs);
      second_fs.close();
      std::cout << "second dict " << second_dict_->nwords() << std::endl;
  

      first_embedding_ = std::make_shared<Matrix>(first_dict_->nwords() + args_->bucket, args_->dim);
      first_embedding_->uniform(1.0 / args_->dim);

      second_embedding_ = std::make_shared<Matrix>(second_dict_->nwords() + args_->bucket, args_->dim);
      second_embedding_->uniform(1.0 / args_->dim);

      first_w1_ = std::make_shared<Matrix>(args_->dim, args_->dim);
      second_w1_ = std::make_shared<Matrix>(args_->dim, args_->dim);

      first_w1_->uniform(1.0 / args_->dim);
      second_w1_->uniform(1.0 / args_->dim);
    }
    model_fs.close();

    std::ifstream train_fs(args_->input);
    if (!train_fs.is_open()) {
      std::cerr << "Input file " << args->input << " cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }

    numToken = 0;
    std::string first, second, label;
    std::vector<std::pair<int32_t, real>> first_words, second_words;
    std::minstd_rand rng(0);
    while (getline(train_fs, first) && getline(train_fs, second) && getline(train_fs, label)) {
      int32_t numToken1 = first_dict_->getLine(first, first_words, rng);
      int32_t numToken2 = second_dict_->getLine(second, second_words, rng);
      if (numToken1 > 30 && numToken2 > 30) numToken += numToken1 + numToken2;
    }
    std::cout << "Total number of token: " << numToken << std::endl;

    start = clock();
    tokenCount = 0;
    real minValidLoss = std::numeric_limits<real>::max();
    for (int32_t epoch = 0; epoch < args_->epoch; epoch++) {
      // train
      std::vector<std::thread> threads;
      for (int32_t i = 0; i < args_->thread; i++) {
        threads.push_back(std::thread([=]() { trainThread(i); }));
      }
      for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
      }
      // valid
      real validLoss = valid();
      std::cout << "\nepoch = " << epoch << "  valid loss = " << validLoss << std::endl;
      if (validLoss < minValidLoss) {
        model_ = std::make_shared<PairModel>(first_embedding_,
                                             first_w1_,
                                             second_embedding_,
                                             second_w1_,
                                             args_, 0);

        saveModel();
        if (args_->model != model_name::sup) {
          saveVectors();
        }
        minValidLoss = std::min(minValidLoss, validLoss);
      }
    }


  }

}
