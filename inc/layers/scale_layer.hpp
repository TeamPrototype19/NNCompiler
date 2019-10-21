#include <iostream>

#include "layer.hpp"

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

    void resizeWeight(int size);
    void resizeBias(int size);
    void setWeight(float val, int index);
    void setBias(float val, int index);

private:
    int _weight_size;
    int _bias_size;
    float *_weight;
    float *_bias;
};

}   // namespace framework
