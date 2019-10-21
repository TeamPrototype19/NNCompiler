#include <iostream>

#include "layer.hpp"
#include "drop_layer.hpp"

using namespace std;
namespace framework {

DropoutLayer::DropoutLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Relu) {
}

DropoutLayer::~DropoutLayer(void) {
}

void DropoutLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string DropoutLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

flatbuffers::Offset<NNFramework::Instruction> 
DropoutLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    return true;
}

}   // namespace framework
