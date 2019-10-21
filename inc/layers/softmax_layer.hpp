#include <iostream>

#include "layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

using namespace std;
namespace framework {

/* Softmax layer class definition
 */
class SoftmaxLayer : public NNLayer {
public:
    SoftmaxLayer(const caffe::LayerParameter& layer_param);
    ~SoftmaxLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

private:
};

}   // namespace framework
