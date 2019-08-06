1. make a library directory
#> mkdir libs

2. Protobuf compile
2.1 Init git submodule (protobuf)
#> git submodule init
Submodule 'protobuf' (https://github.com/protocolbuffers/protobuf.git) registered for path 'protobuf'
#> git submodule update
...
#> cd protobuf
#> git submodule update --init --recursive
2.2 Compile
#> ./autogen.sh
#> ./configure --prefix=/home/deokhwan/Work/GC/NNCompiler/libs
#> make
#> make check
#> make install

3. Compiler compile
#> mkdir build
#> cd build
#> cmake ..
#> make 

4. Run example
#> sh run.sh
