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

int NNLayer::GetInBlobIdx(shared_ptr<Blob> bp) {
    return get_predecessor_idx(bp);
}

int NNLayer::GetOutBlobIdx(shared_ptr<Blob> bp) {
    return get_successor_idx(bp);
}

void NNLayer::SetOutBlobPtr(int i, shared_ptr<Blob> bp) {
    set_successor(i, bp);
}

void NNLayer::SetInBlobPtr(int i, shared_ptr<Blob> bp) {
    set_predecessor(i, bp);
}

/* Get connected previous layer pointers
 */
vector<shared_ptr<NNLayer>> NNLayer::GetPrevConnLayers(void) {
    vector<shared_ptr<NNLayer>> s;

    for(auto bnode : _predecessor) {
        for(auto lnode : bnode->get_predecessor()) {
            auto layer = dynamic_pointer_cast<NNLayer>(lnode);
            assert( layer != nullptr );
            s.push_back( layer );
        }
    }

    return s;
}

/* Get connected next layer pointers
 */
vector<shared_ptr<NNLayer>> NNLayer::GetNextConnLayers(void) {
    vector<shared_ptr<NNLayer>> s;

    for(auto bnode : _successor) {
        for(auto lnode : bnode->get_successor()) {
            auto layer = dynamic_pointer_cast<NNLayer>(lnode);
            assert( layer != nullptr );
            s.push_back( layer );
        }
    }

    return s;
}

/* Drop Layer: 
 * removes this layer and connects its outputs and inputs
 * drop is only possible when 
 * i)   the layer has only 1 input and 1 output blob.
 * ii)  its input and output size are exactly same.
 * iii) prev layer also should have 1-output blob.
 * e.g. Relu, BatchNorm, Scale, etc.
 */
void NNLayer::DropLayer(void) {
    //logfs << "DropLayer is called.\n";
    /* Drop layer condition check
     */
    // condition i: 1-input and 1-output blob
    assert( GetInBlobSize() == GetOutBlobSize() );

    // condition ii: input and output blob size are same.
    auto bp_p = GetInBlobPtr(0);
    auto bp_n = GetOutBlobPtr(0);
    assert( bp_p->isSameSize(bp_n->get_dim()) );

    // condition iii: its prev layer also should have only 1-output blob.
    auto layer_p = GetPrevConnLayers();
    auto layer_n = GetNextConnLayers();
    assert( layer_p.size() == 1 );
    assert( layer_p[0]->GetOutBlobSize() == 1 );
    //assert( layer_n.size() == 1 );
    //assert( layer_n[0]->GetInBlobSize() == 1 );

    /* Drop layer processing
     */
    for(auto next_layer : layer_n) {
        int next_layer_iblob_index = next_layer->GetInBlobIdx(bp_n);
        assert( next_layer_iblob_index >= 0);
        next_layer->SetInBlobPtr(next_layer_iblob_index,  bp_p);
        // TODO: remove layer and blob class instance clearly.
        //logfs << "DROP:: " << _name << "\tnlayer_ib_idx = " << next_layer_iblob_index << "\n";
    }
    bp_p->set_consumer(layer_n);  /// ??????

    return;
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
