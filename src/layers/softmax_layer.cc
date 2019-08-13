#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

SoftmaxLayer::SoftmaxLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Softmax) {
}

SoftmaxLayer::~SoftmaxLayer(void) {
}

void SoftmaxLayer::ComputeOutputSize(void) {
}

string SoftmaxLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
