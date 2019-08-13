#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

FullyConnectedLayer::FullyConnectedLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), FullyConnected) {
}

FullyConnectedLayer::~FullyConnectedLayer(void) {
}

void FullyConnectedLayer::ComputeOutputSize(void) {
}

string FullyConnectedLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
