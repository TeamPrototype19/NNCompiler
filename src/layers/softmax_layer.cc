#include <iostream>

#include "layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

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

flatbuffers::Offset<NNFramework::Instruction> 
SoftmaxLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Softmax OP code generation
     */

    /* Input tile info setting
     */
    auto itiles = setInTileInfo( builder );
    auto otiles = setOutTileInfo( builder );

    auto name = builder.CreateString(_name);
    auto opinfo = NNFramework::CreateSoftmax(builder, name, itiles, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Softmax, 
            NNFramework::OpInfo_Softmax, opinfo.Union() );

    return true;
}

}   // namespace framework
