#include <iostream>

#include "layer.hpp"
#include "blob.hpp"

using namespace std;
namespace framework {

NNLayer::NNLayer(void) {
}

NNLayer::NNLayer(const caffe::LayerParameter& layer_param)
    : Node(layer_param.name(), "layer") {
}

NNLayer::NNLayer(string name, NNLayerType ltype) : Node(name, "layer") {
    _layer_type = ltype;
}

NNLayer::~NNLayer(void) {
}

/* IFM/OFM blob APIs
 */
void NNLayer::add_output_blob( shared_ptr<Blob> bp ) {
    add_successor( static_pointer_cast<Node>(bp) );
}

void NNLayer::add_input_blob( shared_ptr<Blob> bp ) {
    add_predecessor( static_pointer_cast<Node>(bp) );
}

vector<int> NNLayer::get_input_blob_size(int i) {
    if( (unsigned int) i >= _successor.size() )
        throw out_of_range("NNLayer::get_input_blob_size; i over the limit");

    auto blob = dynamic_pointer_cast<Blob>(_successor[i]);
    if( blob == nullptr )
        throw runtime_error("NNLayer::get_input_blob_size; successor is not blob");

    return blob->get_dim();
}

vector<int> NNLayer::get_output_blob_size(int i) {
    if( (unsigned int) i >= _predecessor.size() )
        throw out_of_range("NNLayer::get_input_blob_size: i over the limit");

    auto blob = dynamic_pointer_cast<Blob>(_predecessor[i]);
    if( blob == nullptr )
        throw runtime_error("NNLayer::get_input_blob_size: predecessor is not blob");

    return blob->get_dim();
}

}   // namespace framework
