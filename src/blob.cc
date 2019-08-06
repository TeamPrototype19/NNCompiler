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

}   // namespace framework
