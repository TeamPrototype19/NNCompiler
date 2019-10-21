#include <iostream>

#include "layer.hpp"
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

private:
};

}   // namespace framework
