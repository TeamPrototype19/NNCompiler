#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

FullyConnectedLayer::FullyConnectedLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), FullyConnected) {

    const caffe::InnerProductParameter& param = lparam.inner_product_param();

    _num_output = param.num_output();
}

FullyConnectedLayer::~FullyConnectedLayer(void) {
}

void FullyConnectedLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    vector<int> ob_size = {ib_size[0], _num_output};
    set_output_blob_size(0, ob_size);
}

string FullyConnectedLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
