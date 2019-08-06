#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include <string>

#include "node.hpp"
#include "caffe.pb.h"

using namespace std;

namespace framework {

class Layer : public Node {
public:
    Layer(void);
    Layer(string name);
    Layer(const caffe::LayerParameter& layer_param);
    ~Layer(void);

private:
};

}

#endif
