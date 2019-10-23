#ifndef _BNORM_LAYER_HPP_
#define _BNORM_LAYER_HPP_

#include <iostream>

#include "layer.hpp"
#include "conv_layer.hpp"

using namespace std;
namespace framework {

/* Batch Normalization layer class definition
 */
class BatchNormLayer : public NNLayer {
public:
    BatchNormLayer(const caffe::LayerParameter& layer_param);
    ~BatchNormLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

    void resizeMean(int size);
    void resizeVars(int size);
    void setMean(float val, int index);
    void setVars(float val, int index);
    void setScale(float val);

    void FusingOperation(shared_ptr<ConvLayer> clayer);

private:
    float  _eps;
    bool   _use_global_stats;
    float *_mean;
    float *_vars;
    float  _scale;
    int    _mean_size;
    int    _vars_size;
};

}   // namespace framework
#endif // _BNORM_LAYER_HPP_
