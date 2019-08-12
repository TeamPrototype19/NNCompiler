#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ConvLayer::ConvLayer(const caffe::LayerParameter& layer_param)
    : NNLayer(layer_param.name()) {
}

ConvLayer::~ConvLayer(void) {
}

void ConvLayer::ComputeOutputSize(void) {
}

}   // namespace framework
