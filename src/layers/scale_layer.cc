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
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string ScaleLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
