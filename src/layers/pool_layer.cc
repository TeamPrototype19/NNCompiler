#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

PoolLayer::PoolLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Pooling) {
}

PoolLayer::~PoolLayer(void) {
}

void PoolLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string PoolLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
