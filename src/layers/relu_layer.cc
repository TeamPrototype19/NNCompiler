#include <iostream>

#include "layer.hpp"
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

flatbuffers::Offset<NNExecutor::Instruction> 
ReluLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Relu OP code generation
     */

    /* Input tile info setting
     */
    auto ibp = GetInBlobPtr(0);
    unsigned long iaddr = ibp->get_mem_addr();
    int its_n = ibp->get_dim()[N];
    int its_c = ibp->get_dim()[C];
    int its_h = ibp->get_dim()[H];
    int its_w = ibp->get_dim()[W];
    auto itinfo = NNExecutor::CreateTileInfo( builder, 
            iaddr, its_n, its_c, its_h, its_w );

    auto obp = GetOutBlobPtr(0);
    unsigned long oaddr = obp->get_mem_addr();
    int ots_n = obp->get_dim()[N];
    int ots_c = obp->get_dim()[C];
    int ots_h = obp->get_dim()[H];
    int ots_w = obp->get_dim()[W];
    auto otinfo = NNExecutor::CreateTileInfo( builder, 
            oaddr, ots_n, ots_c, ots_h, ots_w );

    std::vector<flatbuffers::Offset<NNExecutor::TileInfo>> itinfo_vector;
    itinfo_vector.push_back( itinfo );
    auto itiles = builder.CreateVector( itinfo_vector );

    std::vector<flatbuffers::Offset<NNExecutor::TileInfo>> otinfo_vector;
    otinfo_vector.push_back( otinfo );
    auto otiles = builder.CreateVector( otinfo_vector );

    auto name = builder.CreateString(_name);
    auto opinfo = NNExecutor::CreateRelu(builder, name, itiles, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNExecutor::OpCode_Relu, 
            NNExecutor::OpInfo_Relu, opinfo.Union() );
}

}   // namespace framework
