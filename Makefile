#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x
OBJS = args.o dictionary.o matrix.o vector.o model.o utils.o fasttext.o pairmodel.o pairtext.o interplatemodel.o interplatetext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext
opt: pairtext
opt: interplate

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext
debug: pairtext
debug: interplate

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

fasttext: $(OBJS) src/fasttext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext

pairmodel.o: src/pairmodel.cc src/pairmodel.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/pairmodel.cc

pairtext.o: src/pairtext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/pairtext.cc

pairtext: $(OBJS) src/pairtext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/pairmain.cc -o pairtext

interplatemodel.o: src/InterplateModel.cc src/InterplateModel.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/InterplateModel.cc
interplatetext.o: src/InterplateText.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/InterplateText.cc
interplate: $(OBJS) src/InterplateText.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/InterplateMain.cc -o interplate

clean:
	rm -rf *.o fasttext pairtext interplate
