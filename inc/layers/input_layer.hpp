#include <iostream>

#include "layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

using namespace std;
namespace framework {

/* Input layer class definition
 */
class InputLayer : public NNLayer {
public:
    InputLayer(const caffe::LayerParameter& layer_param);
    ~InputLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

private:
    vector<int> _dim;
};

}   // namespace framework
