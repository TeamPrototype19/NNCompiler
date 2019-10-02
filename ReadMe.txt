1. make a library directory
#> mkdir libs

2. Protobuf & Flatbuffer compile
2.1 Init git submodule (protobuf, Flatbuffers)
#> git submodule init
Submodule 'protobuf' (https://github.com/protocolbuffers/protobuf.git) registered for path 'protobuf'
Submodule 'flatbuffers' (https://github.com/google/flatbuffers.git) registered for path 'flatbuffers'
#> git submodule update

2.1 Protobuf compile
#> cd protobuf
#> git submodule update --init --recursive
#> ./autogen.sh
#> ./configure --prefix={"absolute path to"}/NNCompiler/libs
#> make
#> make check
#> make install
#> cd ..

2.2 Flatbuffer compile
#> cd flatbuffers
#> mkdir build
#> cd build
#> cmake ..
#> make flatc
#> cd ..

3. Compiler compile
#> mkdir build
#> cd build
#> cmake ..
#> make 

4. Run example
#> sh run.sh
