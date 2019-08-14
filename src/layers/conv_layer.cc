#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ConvLayer::ConvLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Convolution) {
}

ConvLayer::~ConvLayer(void) {
}

void ConvLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string ConvLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
