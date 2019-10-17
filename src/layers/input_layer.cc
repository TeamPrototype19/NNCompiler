#include <iostream>

#include "layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

using namespace std;
namespace framework {

InputLayer::InputLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Input) {
    if( lparam.has_input_param() ) {
        // TODO: shape can have multiple items.
        // currently, only 1 item can be supported.
        for(int i = 0; i < lparam.input_param().shape(0).dim_size(); i++)
            _dim.push_back(lparam.input_param().shape(0).dim(i));
    }
    else
        throw runtime_error("InputLayer::InputLayer; there is no blob size info!");
}

InputLayer::~InputLayer(void) {
}

void InputLayer::ComputeOutputSize(void) {
    set_output_blob_size(0, _dim);
}

string InputLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

flatbuffers::Offset<NNFramework::Instruction> 
InputLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Input tile info setting
     */
    auto otiles = setOutTileInfo( builder );

    auto name = builder.CreateString(_name);
    auto opinfo = NNFramework::CreateInput(builder, name, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Input, 
            NNFramework::OpInfo_Input, opinfo.Union() );
}

}   // namespace framework
