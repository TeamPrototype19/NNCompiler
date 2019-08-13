#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

PoolLayer::PoolLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Pooling) {
}

PoolLayer::~PoolLayer(void) {
}

void PoolLayer::ComputeOutputSize(void) {
}

string PoolLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
