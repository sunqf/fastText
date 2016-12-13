cmake_minimum_required(VERSION 3.6)
project(fastText)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
    src/args.cc
    src/args.h
    src/dictionary.cc
    src/dictionary.h
    src/fasttext.cc
    src/fasttext.h
    src/main.cc
    src/matrix.cc
    src/matrix.h
    src/model.cc
    src/model.h
    src/real.h
    src/utils.cc
    src/utils.h
    src/vector.cc
    src/vector.h
    src/pairtext.h
    src/pairtext.cc
    src/pairmodel.h
    src/pairmodel.cc)

add_executable(fastText ${SOURCE_FILES} src/pairmain.cc)
add_executable(pairText ${SOURCE_FILES})