#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

FullyConnectedLayer::FullyConnectedLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), FullyConnected) {
}

FullyConnectedLayer::~FullyConnectedLayer(void) {
}

void FullyConnectedLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string FullyConnectedLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
