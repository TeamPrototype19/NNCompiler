#include <iostream>

#include "layer.hpp"

using namespace std;
namespace framework {

Layer::Layer(void) {
}

Layer::Layer(const caffe::LayerParameter& layer_param)
    : Node(layer_param.name(), "layer") {
}

Layer::Layer(string name) : Node(name, "layer") {
}

Layer::~Layer(void) {
}

}   // namespace framework
