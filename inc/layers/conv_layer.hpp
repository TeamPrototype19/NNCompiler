#ifndef _CONV_LAYER_HPP_
#define _CONV_LAYER_HPP_

#include <iostream>

#include "layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"
#include <string>

using namespace std;
namespace framework {

/* Convolution layer class definition
 */
class ConvLayer : public NNLayer {
public:
    ConvLayer(const caffe::LayerParameter& layer_param);
    ~ConvLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

    void resizeWeight(int size);
    void resizeBias(int size);
    void setWeight(float val, int index);
    void setBias(float val, int index);
    float getWeight(int index);
    float getBias(int index);
    int getWeightSize(void);
    int getBiasSize(void);
    void initBiasZero(void);
    void setReluOpEn(bool en);

private:
    int _weight_size;
    int _bias_size;
    float *_weight;
    float *_bias;
    int _kernel_w, _kernel_h;
    int _stride_w, _stride_h;
    int _pad_w, _pad_h;
    int _group;
    int _num_output;
    bool _relu_op_en;
};

}   // namespace framework
#endif  // _CONV_LAYER_HPP_
