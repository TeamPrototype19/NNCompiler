#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

PoolLayer::PoolLayer(const caffe::LayerParameter& layer_param)
    : NNLayer(layer_param.name()) {
}

PoolLayer::~PoolLayer(void) {
}

void PoolLayer::ComputeOutputSize(void) {
}

}   // namespace framework
