#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

PoolLayer::PoolLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Pooling) {

    const caffe::PoolingParameter& param = lparam.pooling_param();

    if( param.has_pad_w() ) _pad_w = param.pad_w();
    else                    _pad_w = 0;
    if( param.has_pad_h() ) _pad_h = param.pad_h();
    else                    _pad_h = 0;
    if( param.has_stride_w() ) _stride_w = param.stride_w();
    else                       _stride_w = 1;
    if( param.has_stride_h() ) _stride_h = param.stride_h();
    else                       _stride_h = 1;
    if( param.has_kernel_w() ) _kernel_w = param.kernel_w();
    else                       _kernel_w = 1;
    if( param.has_kernel_h() ) _kernel_h = param.kernel_h();
    else                       _kernel_h = 1;

    if( param.has_global_pooling() )
        _global_pooling = param.global_pooling();
    else
        _global_pooling = false;

    if( param.has_pool() ) {
        if( param.pool() == caffe::PoolingParameter_PoolMethod_MAX )
            _pool_type = MAX_POOL;
        else if( param.pool() == caffe::PoolingParameter_PoolMethod_AVE )
            _pool_type = AVE_POOL;
        else if( param.pool() == caffe::PoolingParameter_PoolMethod_STOCHASTIC )
            _pool_type = STOCHASTIC_POOL;
        else
            throw runtime_error("Unknown pooling type!");
    }

    if( param.has_kernel_size() ) {
        _kernel_w = param.kernel_size();
        _kernel_h = param.kernel_size();
    }
    if( param.has_stride() ) {
        _stride_w = param.stride();
        _stride_h = param.stride();
    }


    if( LOG_LEVEL >= 2 ) {
        logfs << "Read layer result -------------------\n";
        logfs << "name = " << _name << "\n";
        logfs << "type = " << ltype2str[ _layer_type ] << "\n";
        logfs << "+ internal info\n";
        logfs << "  kernel_w = " << _kernel_w << "\n";
        logfs << "  kernel_h = " << _kernel_h << "\n";
        logfs << "  stride_w = " << _stride_w << "\n";
        logfs << "  stride_h = " << _stride_h << "\n";
        logfs << "  pad_w    = " << _pad_w    << "\n";
        logfs << "  pad_h    = " << _pad_h    << "\n";
        logfs << "  pool_type= " << _pool_type<< "\n";
        logfs << "  glb_pool = " << (_global_pooling ? 1 : 0) << "\n";
        logfs << "\n";
    }

}

PoolLayer::~PoolLayer(void) {
}

void PoolLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    set_output_blob_size(0, ib_size);
}

string PoolLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
