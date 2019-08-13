#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ReluLayer::ReluLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Relu) {
}

ReluLayer::~ReluLayer(void) {
}

void ReluLayer::ComputeOutputSize(void) {
}

string ReluLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
