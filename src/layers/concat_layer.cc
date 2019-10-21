#include <iostream>

#include "layer.hpp"
#include "concat_layer.hpp"

using namespace std;
namespace framework {

ConcatLayer::ConcatLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Concat) {

    const caffe::ConcatParameter& param = lparam.concat_param();

    if( param.has_axis() )
        axis = param.axis();
    else
        axis = 1;

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
    assert( axis < (int) ib_size.size() );
    int ib_size_ch = ib_size[ axis ];

    for(int i = 1; i < get_indegree(); i++) {
        vector<int> ib_size2 = get_input_blob_size(i);
        ib_size_ch += ib_size2[ axis ];

        // size check
        for(int j = 0; j < (int)ib_size.size(); j++) {
            if( j != axis ) 
                assert( ib_size[j] == ib_size2[j] );
        }
    }

    vector<int> ob_size = ib_size;
    ob_size[ axis ] = ib_size_ch;
    set_output_blob_size(0, ob_size);
}

string ConcatLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

flatbuffers::Offset<NNFramework::Instruction> 
ConcatLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Concat OP code generation
     */
    auto name = builder.CreateString(_name);

    /* Input tile info setting
     */
    auto itiles = setInTileInfo( builder );
    auto otiles = setOutTileInfo( builder );

    /* Create Concat table structure 
     */
    auto opinfo = NNFramework::CreateConcat(builder, name, itiles, otiles );

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Concat, 
            NNFramework::OpInfo_Concat, opinfo.Union() );
}

}   // namespace framework
