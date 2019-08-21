#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

ConvLayer::ConvLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Convolution) {

    const caffe::ConvolutionParameter& param = lparam.convolution_param();

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

    if( param.kernel_size_size() > 0 ) {
        _kernel_w = param.kernel_size(0);
        if( param.kernel_size_size() == 1 )
            _kernel_h = param.kernel_size(0);
        else if( param.kernel_size_size() == 2 )
            _kernel_h = param.kernel_size(1);
        else
            throw runtime_error("Doesn't support ConvolutionParameter::kernel_size_size() > 2.");
    }
    if( param.stride_size() > 0 ) {
        _stride_w = param.stride(0);
        if( param.stride_size() == 1 )
            _stride_h = param.stride(0);
        else if( param.stride_size() == 2 )
            _stride_h = param.stride(1);
        else
            throw runtime_error("Doesn't support ConvolutionParameter::stride_size() > 2.");
    }
    if( param.pad_size() > 0 ) {
        _pad_w = param.pad(0);
        if( param.pad_size() == 1 )
            _pad_h = param.pad(0);
        else if( param.pad_size() == 2 )
            _pad_h = param.pad(1);
        else
            throw runtime_error("Doesn't support ConvolutionParameter::pad_size() > 2.");
    }

    if( param.has_group() )
        _group = param.group();

    _num_output = param.num_output();

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
        logfs << "  group    = " << _group    << "\n";
        logfs << "  out_num  = " << _num_output << "\n";
        logfs << "\n";
    }
}

ConvLayer::~ConvLayer(void) {
}

void ConvLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    assert( ib_size.size() == 4 );

    // Caffe uses [N=0,C,H,W] dimension representation.
    int ow = (ib_size[3] + 2*_pad_w - _kernel_w) / _stride_w + 1;
    int oh = (ib_size[2] + 2*_pad_w - _kernel_w) / _stride_w + 1;
    int oc = _num_output;

    vector<int> ob_size = {ib_size[0], oc, oh, ow};
    set_output_blob_size(0, ob_size);

    if( LOG_LEVEL >= 2 ) {
        logfs << "layer.name = " << _name << "\n";
        logfs << "+ IFM.size = [";
        logfs << ib_size[0] << "," << ib_size[1] << ",";
        logfs << ib_size[2] << "," << ib_size[3] << "]\n";
        logfs << "+ OFM.size = [";
        logfs << ob_size[0] << "," << ob_size[1] << ",";
        logfs << ob_size[2] << "," << ob_size[3] << "]\n";
    }
}

string ConvLayer::getLayerInfoStr(void) {
    return " (" + ltype2str[ _layer_type ] + ") ";
}

}   // namespace framework
