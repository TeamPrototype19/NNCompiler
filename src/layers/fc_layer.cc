#include <iostream>

#include "layer.hpp"
#include "blob.hpp"

using namespace std;
namespace framework {

FullyConnectedLayer::FullyConnectedLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), FullyConnected) {

    const caffe::InnerProductParameter& param = lparam.inner_product_param();

    _weight = nullptr;
    _bias = nullptr;

    _num_output = param.num_output();
}

FullyConnectedLayer::~FullyConnectedLayer(void) {
    if( _weight != nullptr )
        delete [] _weight;
    if( _bias != nullptr )
        delete [] _bias;
}

void FullyConnectedLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    vector<int> ob_size = {ib_size[0], _num_output};
    set_output_blob_size(0, ob_size);
}

string FullyConnectedLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

flatbuffers::Offset<NNFramework::Instruction> 
FullyConnectedLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* FullyConnected OP code generation
     */

    /* (Input/Output) Tile info setting
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
    auto opinfo = NNFramework::CreateFC(builder, name, 
            weight, bias, itiles, otiles );


    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_FullyConnected, 
            NNFramework::OpInfo_FC, opinfo.Union() );
}

void FullyConnectedLayer::resizeWeight(int size) {
    if( _weight != nullptr )
        delete [] _weight;
    _weight = new float[ size ];
    _weight_size = size;
}

void FullyConnectedLayer::resizeBias(int size) {
    if( _bias != nullptr )
        delete [] _bias;
    _bias = new float[ size ];
    _bias_size = size;
}

void FullyConnectedLayer::setWeight(float val, int index) {
    assert( index < _weight_size );
    _weight[ index ] = val;
}

void FullyConnectedLayer::setBias(float val, int index) {
    assert( index < _bias_size );
    _bias[ index ] = val;
}

}   // namespace framework
