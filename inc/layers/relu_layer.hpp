#ifndef _RELU_LAYER_HPP_
#define _RELU_LAYER_HPP_


#include <iostream>

#include "layer.hpp"
#include "conv_layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

using namespace std;
namespace framework {

/* Relu layer class definition
 */
class ReluLayer : public NNLayer {
public:
    ReluLayer(const caffe::LayerParameter& layer_param);
    ~ReluLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

    void FusingOperation(shared_ptr<ConvLayer> clayer);

private:
};

}   // namespace framework
#endif // _RELU_LAYER_HPP_
