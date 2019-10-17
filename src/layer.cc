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

int NNLayer::GetOutBlobSize(void) {
    return get_outdegree();
}

int NNLayer::GetInBlobSize(void) {
    return get_indegree();
}

shared_ptr<Blob> NNLayer::GetOutBlobPtr(int i) {
    assert( i < get_outdegree() );
    return dynamic_pointer_cast<Blob>(_successor[i]);
}

shared_ptr<Blob> NNLayer::GetInBlobPtr(int i) {
    assert( i < get_indegree() );
    return dynamic_pointer_cast<Blob>(_predecessor[i]);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NNFramework::TileInfo>>>
NNLayer::setInTileInfo(flatbuffers::FlatBufferBuilder &builder) {
    /* Input tile info setting
     */
    std::vector<flatbuffers::Offset<NNFramework::TileInfo>> itinfo_vector;

    for(int i = 0 ; i < GetInBlobSize() ; i++) {
        auto ibp = GetInBlobPtr(i);
        unsigned long iaddr = ibp->get_mem_addr();
        auto tsize = builder.CreateVector( ibp->get_dim() );
        auto itinfo = NNFramework::CreateTileInfo( builder, iaddr, tsize );
        itinfo_vector.push_back( itinfo );
    }

    return builder.CreateVector( itinfo_vector );
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NNFramework::TileInfo>>>
NNLayer::setOutTileInfo(flatbuffers::FlatBufferBuilder &builder) {
    /* Output tile info setting
     */
    std::vector<flatbuffers::Offset<NNFramework::TileInfo>> otinfo_vector;

    for(int i = 0 ; i < GetOutBlobSize() ; i++) {
        auto obp = GetOutBlobPtr(i);
        unsigned long oaddr = obp->get_mem_addr();
        auto tsize = builder.CreateVector( obp->get_dim() );
        auto otinfo = NNFramework::CreateTileInfo( builder, oaddr, tsize );
        otinfo_vector.push_back( otinfo );
    }

    return builder.CreateVector( otinfo_vector );
}

}   // namespace framework
