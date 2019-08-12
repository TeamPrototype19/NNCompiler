#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include <string>
#include <memory>

#include "node.hpp"
#include "caffe.pb.h"

using namespace std;

namespace framework {

class Blob;

class NNLayer : public Node {
public:
    NNLayer(void);
    NNLayer(string name);
    NNLayer(const caffe::LayerParameter& layer_param);
    ~NNLayer(void);

    void add_output_blob(shared_ptr<Blob> bp);
    void add_input_blob(shared_ptr<Blob> bp);
    virtual void ComputeOutputSize(void) {}

private:
    vector<int> get_input_blob_size(int i);
    vector<int> get_output_blob_size(int i);
};


/* Convolution layer class definition
 */
class ConvLayer : public NNLayer {
public:
    ConvLayer(const caffe::LayerParameter& layer_param);
    ~ConvLayer(void);
    virtual void ComputeOutputSize(void) override;

private:
    int _kernel_w, _kernel_h;
    int _stride_w, _stride_h;
    int _pad_w, _pad_h;
    int group;
};


/* Pooling layer class definition
 */
class PoolLayer : public NNLayer {
    enum PoolType {
        AVG_POOL,
        MAX_POOL
    };
public:
    PoolLayer(const caffe::LayerParameter& layer_param);
    ~PoolLayer(void);
    virtual void ComputeOutputSize(void) override;

private:
    int _kernel_w, _kernel_h;
    int _stride_w, _stride_h;
    int _pad_w, _pad_h;
    bool _global_pool;
    PoolType _pool_type;
};


/* Concat layer class definition
 */
class ConcatLayer : public NNLayer {
public:
    ConcatLayer(const caffe::LayerParameter& layer_param);
    ~ConcatLayer(void);
    virtual void ComputeOutputSize(void) override;

private:
};


/* Fully connected layer class definition
 */
class FullyConnectedLayer : public NNLayer {
public:
    FullyConnectedLayer(const caffe::LayerParameter& layer_param);
    ~FullyConnectedLayer(void);
    virtual void ComputeOutputSize(void) override;

private:
};

}

#endif
