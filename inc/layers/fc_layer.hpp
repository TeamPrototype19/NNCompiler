#ifndef _FC_LAYER_HPP_
#define _FC_LAYER_HPP_

#include <iostream>

#include "layer.hpp"
#include "blob.hpp"

using namespace std;
namespace framework {

/* Fully connected layer class definition
 */
class FullyConnectedLayer : public NNLayer {
public:
    FullyConnectedLayer(const caffe::LayerParameter& layer_param);
    ~FullyConnectedLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

    void resizeWeight(int size);
    void resizeBias(int size);
    void setWeight(float val, int index);
    void setBias(float val, int index);

private:
    int _weight_size;
    int _bias_size;
    float *_weight;
    float *_bias;
    int _num_output;
};

}   // namespace framework
#endif // _FC_LAYER_HPP_
