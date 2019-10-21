#include <iostream>

#include "layer.hpp"
#include "blob.hpp"
#include "instPacket_generated.h"

using namespace std;
namespace framework {

/* Pooling layer class definition
 */
class PoolLayer : public NNLayer {
    enum PoolType {
        MAX_POOL,
        AVE_POOL,
        STOCHASTIC_POOL
    };
public:
    PoolLayer(const caffe::LayerParameter& layer_param);
    ~PoolLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

private:
    int _kernel_w, _kernel_h;
    int _stride_w, _stride_h;
    int _pad_w, _pad_h;
    bool _global_pooling;
    PoolType _pool_type;
};

}   // namespace framework
