#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ConvLayer::ConvLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Convolution) {
}

ConvLayer::~ConvLayer(void) {
}

void ConvLayer::ComputeOutputSize(void) {
}

string ConvLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
