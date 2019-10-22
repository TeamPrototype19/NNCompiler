#include <iostream>

#include "blob.hpp"
#include "layer.hpp"

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

vector<int> Blob::get_dim(void) {
    return _dim;
}

void Blob::add_producer(shared_ptr<NNLayer> lp) {
    add_predecessor( static_pointer_cast<Node>(lp) );
}

void Blob::add_consumer(shared_ptr<NNLayer> lp) {
    add_successor( static_pointer_cast<Node>(lp) );
}

void Blob::set_mem_addr(unsigned long addr) {
    _addr = addr;
}

void Blob::set_producer(int i, shared_ptr<NNLayer> lp) {
    set_predecessor( i, static_pointer_cast<Node>(lp) );
}

void Blob::set_consumer(int i, shared_ptr<NNLayer> lp) {
    set_successor( i, static_pointer_cast<Node>(lp) );
}

unsigned long Blob::get_mem_addr(void) {
    return _addr;
}

bool Blob::isSameSize(vector<int> dim) {
    if( _dim.size() != dim.size() )
        return false;
    for(unsigned int i = 0; i < _dim.size(); i++) {
        if( _dim[i] != _dim[i] )
            return false;
    }
    return true;
}

string Blob::getSizeInfoStr(void) {
    string a;

    if( _dim.size() == 0 )
        return a;

    a += "[";
    int i = 0;
    for(; i < ((int)_dim.size())-1; i++) {
        a += to_string(_dim[i]) + ",";
    }
    a += to_string(_dim[i]) + "]";
    return a;
}

}   // namespace framework
