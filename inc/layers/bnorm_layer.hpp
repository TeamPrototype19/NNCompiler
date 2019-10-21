#include <iostream>

#include "layer.hpp"

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

private:
};

}   // namespace framework
