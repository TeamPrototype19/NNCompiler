#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

BatchNormLayer::BatchNormLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), BatchNorm) {
}

BatchNormLayer::~BatchNormLayer(void) {
}

void BatchNormLayer::ComputeOutputSize(void) {
}

string BatchNormLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
