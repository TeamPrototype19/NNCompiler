#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

FullyConnectedLayer::FullyConnectedLayer(const caffe::LayerParameter& layer_param)
    : NNLayer(layer_param.name()) {
}

FullyConnectedLayer::~FullyConnectedLayer(void) {
}

void FullyConnectedLayer::ComputeOutputSize(void) {
}

}   // namespace framework
