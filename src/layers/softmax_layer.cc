#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

SoftmaxLayer::SoftmaxLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Softmax) {
}

SoftmaxLayer::~SoftmaxLayer(void) {
}

void SoftmaxLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string SoftmaxLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

flatbuffers::Offset<NNExecutor::Instruction> 
SoftmaxLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    return true;
}

}   // namespace framework
