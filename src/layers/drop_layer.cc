#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

DropoutLayer::DropoutLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Relu) {
}

DropoutLayer::~DropoutLayer(void) {
}

void DropoutLayer::ComputeOutputSize(void) {
}

string DropoutLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
