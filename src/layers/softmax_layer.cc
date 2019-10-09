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
    auto ibp = GetInBlobPtr(0);
    unsigned long iaddr = ibp->get_mem_addr();
    int its_n = ibp->get_dim()[N];
    int its_c = ibp->get_dim()[C];
    int its_h = ibp->get_dim()[H];
    int its_w = ibp->get_dim()[W];
    auto itinfo = NNFramework::CreateTileInfo( builder, 
            iaddr, its_n, its_c, its_h, its_w );

    auto obp = GetOutBlobPtr(0);
    unsigned long oaddr = obp->get_mem_addr();
    int ots_n = obp->get_dim()[N];
    int ots_c = obp->get_dim()[C];
    int ots_h = obp->get_dim()[H];
    int ots_w = obp->get_dim()[W];
    auto otinfo = NNFramework::CreateTileInfo( builder, 
            oaddr, ots_n, ots_c, ots_h, ots_w );

    std::vector<flatbuffers::Offset<NNFramework::TileInfo>> itinfo_vector;
    itinfo_vector.push_back( itinfo );
    auto itiles = builder.CreateVector( itinfo_vector );

    std::vector<flatbuffers::Offset<NNFramework::TileInfo>> otinfo_vector;
    otinfo_vector.push_back( otinfo );
    auto otiles = builder.CreateVector( otinfo_vector );

    auto name = builder.CreateString(_name);
    auto opinfo = NNFramework::CreateSoftmax(builder, name, itiles, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Softmax, 
            NNFramework::OpInfo_Softmax, opinfo.Union() );

    return true;
}

}   // namespace framework
