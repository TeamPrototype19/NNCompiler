#include <iostream>

#include "layer.hpp"
#include "scale_layer.hpp"

using namespace std;
namespace framework {

ScaleLayer::ScaleLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Scale) {

    _weight = nullptr;
    _bias = nullptr;
    _weight_size = 0;
    _bias_size = 0;

    if( LOG_LEVEL >= 2 ) {
        logfs << "Read layer result -------------------\n";
        logfs << "name = " << _name << "\n";
        logfs << "type = " << ltype2str[ _layer_type ] << "\n";
        logfs << "\n";
    }
}

ScaleLayer::~ScaleLayer(void) {
    if( _weight != nullptr )
        delete [] _weight;
    if( _bias != nullptr )
        delete [] _bias;
}

void ScaleLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string ScaleLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

flatbuffers::Offset<NNFramework::Instruction> 
ScaleLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Scale OP code generation
     */

    /* Input tile info setting
     */
    auto itiles = setInTileInfo( builder );
    auto otiles = setOutTileInfo( builder );

    /* Weight & Bias array setting 
     */
    auto name = builder.CreateString(_name);
    auto weight = builder.CreateVector( _weight, _weight_size );
    auto bias   = builder.CreateVector( _bias  , _bias_size   );

    /* Create Conv table structure 
     */
    auto opinfo = NNFramework::CreateScale(builder, name, weight, bias, itiles, otiles );

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Scale, 
            NNFramework::OpInfo_Scale, opinfo.Union() );
}

void ScaleLayer::resizeWeight(int size) {
    if( _weight != nullptr )
        delete [] _weight;
    _weight = new float[ size ];
    _weight_size = size;
}

void ScaleLayer::resizeBias(int size) {
    if( _bias != nullptr )
        delete [] _bias;
    _bias = new float[ size ];
    _bias_size = size;
}

void ScaleLayer::setWeight(float val, int index) {
    assert( index < _weight_size );
    _weight[ index ] = val;
}

void ScaleLayer::setBias(float val, int index) {
    assert( index < _bias_size );
    _bias[ index ] = val;
}


}   // namespace framework
