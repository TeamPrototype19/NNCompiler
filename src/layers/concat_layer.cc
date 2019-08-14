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
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string ConcatLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
