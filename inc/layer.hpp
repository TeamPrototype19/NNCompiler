#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include <string>
#include <memory>

#include "log.h"
#include "node.hpp"
#include "caffe.pb.h"
#include "instPacket_generated.h"

using namespace std;

namespace framework {

class Blob;

enum NNLayerType {
    Input = 0,
    Convolution,
    Relu,
    Pooling,
    Concat,
    Scale,
    Softmax,
    FullyConnected,
    BatchNorm,
    Dropout
};

class NNLayer : public Node {
public:
    NNLayer(void);
    NNLayer(string name, NNLayerType ltype);
    NNLayer(const caffe::LayerParameter& layer_param);
    ~NNLayer(void);

    NNLayerType get_layer_type();
    string get_layer_type_str();
    void add_output_blob(shared_ptr<Blob> bp);
    void add_input_blob(shared_ptr<Blob> bp);
    int  GetInBlobSize(void);
    int  GetOutBlobSize(void);
    shared_ptr<Blob> GetInBlobPtr(int i);
    shared_ptr<Blob> GetOutBlobPtr(int i);

    virtual void ComputeOutputSize(void) = 0;
    virtual string getLayerInfoStr(void) = 0;
    virtual flatbuffers::Offset<NNFramework::Instruction> 
        GenerateCompiledOutput(flatbuffers::FlatBufferBuilder &builder) = 0;

    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NNFramework::TileInfo>>>
    setInTileInfo(flatbuffers::FlatBufferBuilder &builder);
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NNFramework::TileInfo>>>
    setOutTileInfo(flatbuffers::FlatBufferBuilder &builder);

    map<NNLayerType, string> ltype2str = {
        {Input         , "Input"},
        {Convolution   , "Convolution"},
        {Relu          , "ReLU"},
        {Pooling       , "Pooling"},
        {Concat        , "Concat"},
        {Scale         , "Scale"},
        {Softmax       , "Softmax"},
        {FullyConnected, "FullyConnected"},
        {BatchNorm     , "BatchNorm"},
        {Dropout       , "DropOut"}
    };

protected:
    vector<int> get_input_blob_size(int i);
    vector<int> get_output_blob_size(int i);
    void set_output_blob_size(int i, vector<int>);

    NNLayerType _layer_type;
};

}

#endif
