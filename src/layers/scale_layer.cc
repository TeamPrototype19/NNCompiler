#include <iostream>

#include "layer.hpp"
#include "scale_layer.hpp"

using namespace std;
namespace framework {

ScaleLayer::ScaleLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Scale) {

    _scale = nullptr;
    _bias = nullptr;
    _scale_size = 0;
    _bias_size = 0;

    if( LOG_LEVEL >= 2 ) {
        logfs << "Read layer result -------------------\n";
        logfs << "name = " << _name << "\n";
        logfs << "type = " << ltype2str[ _layer_type ] << "\n";
        logfs << "\n";
    }
}

ScaleLayer::~ScaleLayer(void) {
    if( _scale != nullptr )
        delete [] _scale;
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

void ScaleLayer::FusingOperation(shared_ptr<ConvLayer> clayer) {
    /* Fusing the scale and bias with weight and bias of Convolution.
     */
    int conv_weight_size = clayer->getWeightSize();
    int ch1_weight_size  = conv_weight_size / _scale_size;
    for(int i = 0; i < _scale_size; i++) {
        for(int j = 0; j < ch1_weight_size; j++) {
            int widx = i*ch1_weight_size + j;
            clayer->setWeight((clayer->getWeight(widx) / _scale[i]), widx);
        }
    }

    if( _bias_size > 0 ) {
        int conv_bias_size   = clayer->getBiasSize();
        assert( conv_bias_size == _bias_size );
        assert( _scale_size == _bias_size );
        for(int i = 0; i < _bias_size; i++) {
            clayer->setBias(((clayer->getBias(i) * _scale[i]) + _bias[i]), i);
        }
    }

    return;
}

flatbuffers::Offset<NNFramework::Instruction> 
ScaleLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Scale OP code generation
     */

    /* Input tile info setting
     */
    auto itiles = setInTileInfo( builder );
    auto otiles = setOutTileInfo( builder );

    /* Scale & Bias array setting 
     */
    auto name = builder.CreateString(_name);
    auto scale = builder.CreateVector( _scale , _scale_size );
    auto bias  = builder.CreateVector( _bias  , _bias_size   );

    /* Create Conv table structure 
     */
    auto opinfo = NNFramework::CreateScale(builder, name, scale, bias, itiles, otiles );

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Scale, 
            NNFramework::OpInfo_Scale, opinfo.Union() );
}

void ScaleLayer::resizeScale(int size) {
    if( _scale != nullptr )
        delete [] _scale;
    _scale = new float[ size ];
    _scale_size = size;
}

void ScaleLayer::resizeBias(int size) {
    if( _bias != nullptr )
        delete [] _bias;
    _bias = new float[ size ];
    _bias_size = size;
}

void ScaleLayer::setScale(float val, int index) {
    assert( index < _scale_size );
    _scale[ index ] = val;
}

void ScaleLayer::setBias(float val, int index) {
    assert( index < _bias_size );
    _bias[ index ] = val;
}


}   // namespace framework
