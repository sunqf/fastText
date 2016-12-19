//
// Created by sunqf on 2016/12/15.
//

#include <iostream>

#include "interplatetext.h"
#include "args.h"

using namespace fasttext;

void printUsage() {
  std::cout
      << "usage: fasttext <command> <args>\n\n"
      << "The commands supported by fasttext are:\n\n"
      << "  supervised          train a supervised classifier\n"
      << "  test                evaluate a supervised classifier\n"
      << "  predict             predict most likely labels\n"
      << "  predict-prob        predict most likely labels with probabilities\n"
      << "  print-vectors       print vectors given a trained model\n"
      << std::endl;
}

void printTestUsage() {
  std::cout
      << "usage: fasttext test <model> <test-data> [<k>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << std::endl;
}

void printPredictUsage() {
  std::cout
      << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << std::endl;
}

void printPrintVectorsUsage() {
  std::cout
      << "usage: fasttext print-vectors <model>\n\n"
      << "  <model>      model filename\n"
      << std::endl;
}

void test(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  interplatetext interplateText;
  interplateText.loadModel(std::string(argv[2]));
  std::string infile(argv[3]);
  if (infile == "-") {
    interplateText.test(std::cin);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << __FILE__ << __LINE__ <<  "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    interplateText.test(ifs);
    ifs.close();
  }
  exit(0);
}

void predict(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  bool print_prob = std::string(argv[1]) == "predict-prob";
  interplatetext interplateText;
  interplateText.loadModel(std::string(argv[2]));

  std::string infile(argv[3]);
  if (infile == "-") {
    interplateText.predict(std::cin);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    interplateText.predict(ifs);
    ifs.close();
  }

  exit(0);
}

void printVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintVectorsUsage();
    exit(EXIT_FAILURE);
  }
  interplatetext interplateText;
  interplateText.loadModel(std::string(argv[2]));
  interplateText.printVectors();
  exit(0);
}

void train(int argc, char** argv) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  a->parseArgs(argc, argv);
  interplatetext interplateText;
  interplateText.train(a);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(argv[1]);
  if (command == "supervised") {
    train(argc, argv);
  } else if (command == "test") {
    test(argc, argv);
  } else if (command == "print-vectors") {
    printVectors(argc, argv);
  } else if (command == "predict" || command == "predict-prob" ) {
    predict(argc, argv);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}




