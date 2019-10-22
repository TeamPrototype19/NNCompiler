#include <iostream>

#include "layer.hpp"
#include "bnorm_layer.hpp"

using namespace std;
namespace framework {

BatchNormLayer::BatchNormLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), BatchNorm) {

    const caffe::BatchNormParameter& param = lparam.batch_norm_param();

    _eps = 1.0;
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

flatbuffers::Offset<NNFramework::Instruction> 
BatchNormLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
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

void BatchNormLayer::setEps(float val) {
    _eps = val;
}

}   // namespace framework
