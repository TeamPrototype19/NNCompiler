#include <iostream>

#include "layer.hpp"
#include "pool_layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

using namespace std;
namespace framework {

PoolLayer::PoolLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Pooling) {

    const caffe::PoolingParameter& param = lparam.pooling_param();

    _kernel_w = 1;
    _kernel_h = 1;
    _stride_w = 1;
    _stride_h = 1;
    _pad_w = 0;
    _pad_h = 0;

    if( param.has_pad_w() )
        _pad_w = param.pad_w();
    if( param.has_pad_h() )
        _pad_h = param.pad_h();
    if( param.has_stride_w() )
        _stride_w = param.stride_w();
    if( param.has_stride_h() )
        _stride_h = param.stride_h();
    if( param.has_kernel_w() )
        _kernel_w = param.kernel_w();
    if( param.has_kernel_h() )
        _kernel_h = param.kernel_h();

    if( param.has_kernel_size() ) {
        _kernel_w = param.kernel_size();
        _kernel_h = param.kernel_size();
    }
    if( param.has_stride() ) {
        _stride_w = param.stride();
        _stride_h = param.stride();
    }
    if( param.has_pad() ) {
        _pad_w = param.pad();
        _pad_h = param.pad();
    }

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
    assert( ib_size.size() == 4);

    //dilated(atrous) pooling is not supported currently
    // ow = (ib_size[3] + 2*_pad_w - (_kernel_w*(dilation-1)+1))/_stride_w +1
    int ow = (ib_size[3] + 2*_pad_w - _kernel_w)/_stride_w + 1;
    int oh = (ib_size[2] + 2*_pad_h - _kernel_h)/_stride_h + 1;

    if( _global_pooling ) {
        ow = 1;
        oh = 1;
    }
    
    vector<int> ob_size = {ib_size[0], ib_size[1], oh, ow};
    set_output_blob_size(0, ob_size);

    if( LOG_LEVEL >=2){
        logfs << "layer.name" << _name <<"\n";
        logfs <<  "+ IFM.size=[";
        logfs <<  ib_size[0] << "," << ib_size[1] <<",";
        logfs <<  ib_size[2] << "," << ib_size[3] <<"]\n";
        logfs <<  "+ OFM.size=[";
        logfs <<  ob_size[0] << "," << ob_size[1] <<",";
        logfs <<  ob_size[2] << "," << ob_size[3] << "]\n";
    }
}

string PoolLayer::getLayerInfoStr(void) {
    string msg = " (" + ltype2str[ _layer_type ] + ")";
    msg += "\nk:" + std::to_string(_kernel_w) + "x" + std::to_string(_kernel_h);
    msg += " s:" + std::to_string(_stride_w) + "x" + std::to_string(_stride_h);
    msg += " p:" + std::to_string(_pad_w) + "x" + std::to_string(_pad_h) + " ";

    return msg;
}

flatbuffers::Offset<NNFramework::Instruction> 
PoolLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Pool OP code generation
     */

    /* Input tile info setting
     */
    auto itiles = setInTileInfo( builder );
    auto otiles = setOutTileInfo( builder );

    /* Create Pooling table structure 
     */
    auto name = builder.CreateString(_name);
    auto opinfo = NNFramework::CreatePooling(builder, name, _kernel_w, _kernel_h,
            _stride_w, _stride_h, _pad_w, _pad_h, _global_pooling, itiles, otiles);

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Pooling, 
            NNFramework::OpInfo_Pooling, opinfo.Union() );
    return true;
}

}   // namespace framework
