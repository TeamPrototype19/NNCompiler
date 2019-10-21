#include <iostream>

#include "layer.hpp"
#include "concat_layer.hpp"

using namespace std;
namespace framework {

ConcatLayer::ConcatLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Concat) {

    if( LOG_LEVEL >= 2 ) {
        logfs << "Read layer result -------------------\n";
        logfs << "name = " << _name << "\n";
        logfs << "type = " << ltype2str[ _layer_type ] << "\n";
        logfs << "+ internal info\n";
        logfs << "\n";
    }
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

flatbuffers::Offset<NNFramework::Instruction> 
ConcatLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    return true;
}

}   // namespace framework
