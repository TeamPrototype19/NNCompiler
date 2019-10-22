#include <iostream>

#include "layer.hpp"
#include "conv_layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"
#include <string>

using namespace std;
namespace framework {

ConvLayer::ConvLayer(const caffe::LayerParameter& lparam) 
    : NNLayer(lparam.name(), Convolution) {

    const caffe::ConvolutionParameter& param = lparam.convolution_param();

    _weight = nullptr;
    _bias = nullptr;
    _weight_size = 0;
    _bias_size = 0;
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
    if( _weight != nullptr )
        delete [] _weight;
    if( _bias != nullptr )
        delete [] _bias;
}

void ConvLayer::ComputeOutputSize(void) {
    vector<int> ib_size = get_input_blob_size(0);
    assert( ib_size.size() == 4 );

    // Caffe uses [N=0,C,H,W] dimension representation.
    // dilated(atrous) convolution is not supported currently
    // ow = (ib_size[3] + 2*_pad_w - (_kernel_w*(dilation-1)+1))/_stride_w + 1;
    int ow = (ib_size[3] + 2*_pad_w - _kernel_w) / _stride_w + 1;
    int oh = (ib_size[2] + 2*_pad_h - _kernel_h) / _stride_h + 1;
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
    string msg = " (" + ltype2str[ _layer_type ] + ")";
    msg += "\nk:" + std::to_string(_kernel_w) + "x" + std::to_string(_kernel_h);
    msg += " s:" + std::to_string(_stride_w) + "x" + std::to_string(_stride_h);
    msg += " p:" + std::to_string(_pad_w) + "x" + std::to_string(_pad_h) + " ";

    return msg;
}

flatbuffers::Offset<NNFramework::Instruction> 
ConvLayer::GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) {
    /* Convolution OP code generation
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
    auto opinfo = NNFramework::CreateConv(builder, name, _kernel_w, _kernel_h,
            _stride_w, _stride_h, _pad_w, _pad_h, 
            weight, bias, itiles, otiles );

    /* Generate instruction
     */
    return CreateInstruction( builder, NNFramework::OpCode_Convolution, 
            NNFramework::OpInfo_Conv, opinfo.Union() );
}

void ConvLayer::resizeWeight(int size) {
    if( _weight != nullptr )
        delete [] _weight;
    _weight = new float[ size ];
    _weight_size = size;
}

void ConvLayer::resizeBias(int size) {
    if( _bias != nullptr )
        delete [] _bias;
    _bias = new float[ size ];
    _bias_size = size;
}

void ConvLayer::setWeight(float val, int index) {
    assert( index < _weight_size );
    _weight[ index ] = val;
}

void ConvLayer::setBias(float val, int index) {
    assert( index < _bias_size );
    _bias[ index ] = val;
}

}   // namespace framework
