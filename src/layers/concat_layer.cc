#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ConcatLayer::ConcatLayer(const caffe::LayerParameter& layer_param)
    : NNLayer(layer_param.name()) {
}

ConcatLayer::~ConcatLayer(void) {
}

void ConcatLayer::ComputeOutputSize(void) {
}

}   // namespace framework
