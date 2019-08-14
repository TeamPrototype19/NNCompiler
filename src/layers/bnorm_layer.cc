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
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string BatchNormLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
