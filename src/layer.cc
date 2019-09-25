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

NNLayerType NNLayer::get_layer_type() {
    return _layer_type;
}

string NNLayer::get_layer_type_str() {
    return ltype2str[ _layer_type ];
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
    if( (unsigned int) i >= _predecessor.size() )
        throw out_of_range("NNLayer::get_input_blob_size; i over the limit");

    auto blob = dynamic_pointer_cast<Blob>(_predecessor[i]);
    if( blob == nullptr )
        throw runtime_error("NNLayer::get_input_blob_size; predecessor is not blob");

    return blob->get_dim();
}

vector<int> NNLayer::get_output_blob_size(int i) {
    if( (unsigned int) i >= _successor.size() )
        throw out_of_range("NNLayer::get_output_blob_size: i over the limit");

    auto blob = dynamic_pointer_cast<Blob>(_successor[i]);
    if( blob == nullptr )
        throw runtime_error("NNLayer::get_output_blob_size: successor is not blob");

    return blob->get_dim();
}

void NNLayer::set_output_blob_size(int i, vector<int> b_size) {
    if( (unsigned int) i >= _successor.size() )
        throw out_of_range("NNLayer::get_input_blob_size: i over the limit");

    auto blob = dynamic_pointer_cast<Blob>(_successor[i]);
    if( blob == nullptr )
        throw runtime_error("NNLayer::get_input_blob_size: successor is not blob");

    blob->set_dim(b_size);
}

void NNLayer::SetOutputIndexes(int& index) {
    for(auto node : _successor) {
        auto blob = dynamic_pointer_cast<Blob>(node);
        blob->set_index( index );
        index++;
    }
}

map<shared_ptr<Blob>, vector<int>> NNLayer::GetOutputSize(void) {
    map<shared_ptr<Blob>, vector<int>> bufSize;

    for(auto node : _successor) {
        auto blob = dynamic_pointer_cast<Blob>(node);
        bufSize[blob] = blob->get_dim();
    }

    return bufSize;
}

}   // namespace framework
