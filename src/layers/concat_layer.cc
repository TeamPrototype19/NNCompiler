#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ConcatLayer::ConcatLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Concat) {
}

ConcatLayer::~ConcatLayer(void) {
}

void ConcatLayer::ComputeOutputSize(void) {
}

string ConcatLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
