#ifndef _DROP_LAYER_HPP_
#define _DROP_LAYER_HPP_

#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

/* Drop out layer class definition
 */
class DropoutLayer : public NNLayer {
public:
    DropoutLayer(const caffe::LayerParameter& layer_param);
    ~DropoutLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

private:
};

}   // namespace framework
#endif // _DROP_LAYER_HPP_
