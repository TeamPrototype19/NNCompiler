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
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string ReluLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
