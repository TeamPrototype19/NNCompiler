#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include <string>
#include <memory>

#include "log.h"
#include "node.hpp"
#include "caffe.pb.h"

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
    virtual void ComputeOutputSize(void) = 0;
    virtual string getLayerInfoStr(void) = 0;

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


/* Convolution layer class definition
 */
class ConvLayer : public NNLayer {
public:
    ConvLayer(const caffe::LayerParameter& layer_param);
    ~ConvLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
    int _kernel_w, _kernel_h;
    int _stride_w, _stride_h;
    int _pad_w, _pad_h;
    int _group;
};


/* Input layer class definition
 */
class InputLayer : public NNLayer {
public:
    InputLayer(const caffe::LayerParameter& layer_param);
    ~InputLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
    vector<int> _dim;
};


/* Relu layer class definition
 */
class ReluLayer : public NNLayer {
public:
    ReluLayer(const caffe::LayerParameter& layer_param);
    ~ReluLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


/* Batch Normalization layer class definition
 */
class BatchNormLayer : public NNLayer {
public:
    BatchNormLayer(const caffe::LayerParameter& layer_param);
    ~BatchNormLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


/* Scale layer class definition
 */
class ScaleLayer : public NNLayer {
public:
    ScaleLayer(const caffe::LayerParameter& layer_param);
    ~ScaleLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


/* Drop out layer class definition
 */
class DropoutLayer : public NNLayer {
public:
    DropoutLayer(const caffe::LayerParameter& layer_param);
    ~DropoutLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


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

private:
    int _kernel_w, _kernel_h;
    int _stride_w, _stride_h;
    int _pad_w, _pad_h;
    bool _global_pooling;
    PoolType _pool_type;
};


/* Concat layer class definition
 */
class ConcatLayer : public NNLayer {
public:
    ConcatLayer(const caffe::LayerParameter& layer_param);
    ~ConcatLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


/* Fully connected layer class definition
 */
class FullyConnectedLayer : public NNLayer {
public:
    FullyConnectedLayer(const caffe::LayerParameter& layer_param);
    ~FullyConnectedLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


/* Softmax layer class definition
 */
class SoftmaxLayer : public NNLayer {
public:
    SoftmaxLayer(const caffe::LayerParameter& layer_param);
    ~SoftmaxLayer(void);
    virtual void ComputeOutputSize(void) override;
    virtual string getLayerInfoStr(void) override;

private:
};


}

#endif
