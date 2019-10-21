#include <iostream>

#include "layer.hpp"
#include "relu_layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

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

flatbuffers::Offset<NNFramework::Instruction> 
ReluLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Relu OP code generation
     */

    /* Input tile info setting
     */
    auto itiles = setInTileInfo( builder );
    auto otiles = setOutTileInfo( builder );

    auto name = builder.CreateString(_name);
    auto opinfo = NNFramework::CreateRelu(builder, name, itiles, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Relu, 
            NNFramework::OpInfo_Relu, opinfo.Union() );
}

}   // namespace framework
