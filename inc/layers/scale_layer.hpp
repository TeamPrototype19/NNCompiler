#ifndef _SCALE_LAYER_HPP_
#define _SCALE_LAYER_HPP_

#include <iostream>

#include "layer.hpp"
#include "conv_layer.hpp"

using namespace std;
namespace framework {

/* Scale layer class definition
 */
class ScaleLayer : public NNLayer {
public:
    ScaleLayer(const caffe::LayerParameter& layer_param);
    ~ScaleLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

    void resizeScale(int size);
    void resizeBias(int size);
    void setScale(float val, int index);
    void setBias(float val, int index);

    void FusingOperation(shared_ptr<ConvLayer> clayer);

private:
    int _scale_size;
    int _bias_size;
    float *_scale;
    float *_bias;
};

}   // namespace framework
#endif // _SCALE_LAYER_HPP_
