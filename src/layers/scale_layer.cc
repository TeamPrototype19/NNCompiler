#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ScaleLayer::ScaleLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Scale) {
}

ScaleLayer::~ScaleLayer(void) {
}

void ScaleLayer::ComputeOutputSize(void) {
}

string ScaleLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
