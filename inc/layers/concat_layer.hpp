#ifndef _CONCAT_LAYER_HPP_
#define _CONCAT_LAYER_HPP_

#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

/* Concat layer class definition
 */
class ConcatLayer : public NNLayer {
public:
    ConcatLayer(const caffe::LayerParameter& layer_param);
    ~ConcatLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) override;

private:
    int axis;
};

}   // namespace framework
#endif // _CONCAT_LAYER_HPP_
