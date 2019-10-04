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
    auto obp = GetOutBlobPtr(0);
    unsigned long oaddr = obp->get_mem_addr();
    int ots_n = obp->get_dim()[N];
    int ots_c = obp->get_dim()[C];
    int ots_h = obp->get_dim()[H];
    int ots_w = obp->get_dim()[W];
    auto otinfo = NNFramework::CreateTileInfo( builder, 
            oaddr, ots_n, ots_c, ots_h, ots_w );

    std::vector<flatbuffers::Offset<NNFramework::TileInfo>> otinfo_vector;
    otinfo_vector.push_back( otinfo );
    auto otiles = builder.CreateVector( otinfo_vector );

    auto name = builder.CreateString(_name);
    auto opinfo = NNFramework::CreateInput(builder, name, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Input, 
            NNFramework::OpInfo_Input, opinfo.Union() );
}

}   // namespace framework
