#include <iostream>
#include <cmath>

#include "layer.hpp"
#include "bnorm_layer.hpp"

using namespace std;
namespace framework {

BatchNormLayer::BatchNormLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), BatchNorm) {

    const caffe::BatchNormParameter& param = lparam.batch_norm_param();

    _eps = 0.0;
    _use_global_stats = true;

    if( param.has_eps() )
        _eps = param.eps();
    if( param.has_use_global_stats() )
        _use_global_stats = param.use_global_stats();

    // CHECK: current ver. supports only 'global stat' mode.
    assert( _use_global_stats == true );
}

BatchNormLayer::~BatchNormLayer(void) {
    if( _mean != nullptr )
        delete [] _mean;
    if( _vars != nullptr )
        delete [] _vars;
}

void BatchNormLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string BatchNormLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

void BatchNormLayer::FusingOperation(shared_ptr<ConvLayer> clayer) {
    /* Calcualtes mean and variance values
     */
    float scaleJ = (_scale == 0) ? 0 : (1.0 /_scale);

    for(int i = 0; i < _mean_size; i++)
        _mean[i] = _mean[i] * scaleJ;
    for(int i = 0; i < _vars_size; i++)
        _vars[i] = sqrt(_vars[i] * scaleJ + _eps);

    /* Fusing the mean and variance with weight and bias of Convolution.
     */
    int conv_weight_size = clayer->getWeightSize();
    int ch1_weight_size  = conv_weight_size / _vars_size;
    for(int i = 0; i < _vars_size; i++) {
        for(int j = 0; j < ch1_weight_size; j++) {
            int widx = i*ch1_weight_size + j;
            clayer->setWeight((clayer->getWeight(widx) / _vars[i]), widx);
        }
    }

    if( _mean_size > 0 ) {
        int conv_bias_size   = clayer->getBiasSize();
        assert( conv_bias_size == _mean_size );
        for(int i = 0; i < _mean_size; i++) {
            clayer->setBias(((clayer->getBias(i) - _mean[i]) / _vars[i]), i);
        }
    }

    return;
}

flatbuffers::Offset<NNFramework::Instruction> 
BatchNormLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    throw runtime_error("[ERROR] not support BN layer, yet!");
    return true;
}

void BatchNormLayer::resizeMean(int size) {
    if( _mean != nullptr )
        delete [] _mean;
    _mean = new float[ size ];
    _mean_size = size;
}

void BatchNormLayer::resizeVars(int size) {
    if( _vars != nullptr )
        delete [] _vars;
    _vars = new float[ size ];
    _vars_size = size;
}

void BatchNormLayer::setMean(float val, int index) {
    assert( index < _mean_size );
    _mean[ index ] = val;
}

void BatchNormLayer::setVars(float val, int index) {
    assert( index < _vars_size );
    _vars[ index ] = val;
}

void BatchNormLayer::setScale(float val) {
    _scale = val;
}

}   // namespace framework
