#include <iostream>

#include "blob.hpp"

using namespace std;

namespace framework {

Blob::Blob(void) {
}

Blob::Blob(string name) : Node(name, "blob") {
}

Blob::~Blob(void) {
    _dim.clear();
}

template<typename First, typename... Args>
void Blob::set_dim(First first, Args... args) {
    set_dim(first);
    set_dim(args...);
}

void Blob::set_dim(int arg) {
    _dim.push_back( arg );
}

void Blob::set_dim(std::vector<int> dim) {
    _dim = dim;
}

}   // namespace framework
