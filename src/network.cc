#include <iostream>
#include <fstream>
#include <queue>
#include <stack>

#include "layer.hpp"
#include "blob.hpp"
#include "network.hpp"

using namespace std;

namespace framework {

Network::Network(void) {
}

Network::~Network(void) {
}

Network::Network(const caffe::NetParameter &net, string type) {
    map<string, string> inplace_name;
    map<string, int>    inplace_cnt;


    // set graph
    if( net.has_name() )
        _name = net.name();
    else
        _name = "unknown";


    // set graph type
    if( type.length() > 0 )
        _type = type;
    else
        _type = "unknown";

    vector<int> dim;
    for(int j = 0; j < net.input_dim_size(); j++)
        dim.push_back( net.input_dim(j) );
    input_blob.set_dim( dim );

    for(int i = 0; i < net.input_size(); i++) {
        //string name = net.input(i)+"_node";
        //shared_ptr<NNLayer> p_layer = create_layer( name );
        //_nodes.push_back( p_layer );
        //_name2node.insert( make_pair(p_layer->get_name(), p_layer) );

        shared_ptr<Blob> p_blob = make_shared<Blob>(net.input(i));
        //p_layer->add_output_blob( p_blob );
        //p_blob->add_producer( p_layer );
        _nodes.push_back( p_blob );
        _name2node.insert( make_pair(p_blob->get_name(), p_blob) );
    }


    /* Layer processing
     */
    for(int i = 0; i < net.layer_size(); i++) {
        const caffe::LayerParameter& lparam = net.layer(i);
        shared_ptr<NNLayer> p_layer = create_layer( lparam );
        _nodes.push_back( p_layer );
        _name2node.insert( make_pair(p_layer->get_name(), p_layer) );

        cout << "processing... : " << lparam.name() << endl;

        /* Connection
         */
        for(int j = 0; j < lparam.bottom_size(); j++) {
            string blob_name = lparam.bottom(j);
            if( inplace_name.find(blob_name) != inplace_name.end() )
                blob_name = inplace_name[blob_name];
            //shared_ptr<Node> p_blob = _name2node[blob_name];
            shared_ptr<Blob> p_blob = get_blob_by_name(blob_name);
            p_layer->add_input_blob( p_blob );
            p_blob->add_consumer( p_layer );
        }

        /* Creates blobs and top connection
         */
        for(int j = 0; j < lparam.top_size(); j++) {
            string blob_name = lparam.top(j);

            if( _name2node.find(blob_name) != _name2node.end() ) {
                if( inplace_cnt.find(blob_name) == inplace_cnt.end() )
                    inplace_cnt[blob_name] = 0;

                string new_name;
                do {
                    new_name = blob_name + "_ip" + to_string( inplace_cnt[blob_name] );
                    inplace_cnt[blob_name]++;
                } while( _name2node.find(new_name) != _name2node.end() );

                inplace_name[blob_name] = new_name;
                blob_name = new_name;
            }

            shared_ptr<Blob> p_blob = make_shared<Blob>(blob_name);
            p_layer->add_output_blob( p_blob );
            p_blob->add_producer( p_layer );
            _nodes.push_back( p_blob );
            _name2node.insert( make_pair(p_blob->get_name(), p_blob) );
        }
    }


    /* Find entry node and exit nodes
     */
    for(uint32_t i = 0; i < _nodes.size() ; i++) {
        if( _nodes[i]->get_indegree() == 0 )
            _entry_nodes.push_back( _nodes[i] );
        if( _nodes[i]->get_outdegree() == 0 )
            _exit_nodes.push_back( _nodes[i] );
    }


    /* Output blob size calculation
     */
    sched_layers = ScheduleLayers();

#if 1   // DEBUG
    for(auto layer: sched_layers) {
        cout << "name = " << layer->get_name();
        cout << "\ttype = " << layer->get_layer_type_str() << endl;
    }
#endif
    
    return;
}

shared_ptr<NNLayer> Network::create_layer(const caffe::LayerParameter& lparam) {
    if( ! lparam.has_type() )
        throw runtime_error("Network::create_layer; layer don't have type parameter!");

    string type = lparam.type();
    if( type.compare("Convolution") == 0 ) {
        shared_ptr<ConvLayer> layer = make_shared<ConvLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("ReLU") == 0 ) {
        shared_ptr<ReluLayer> layer = make_shared<ReluLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("Pooling") == 0 ) {
        shared_ptr<PoolLayer> layer = make_shared<PoolLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("InnerProduct") == 0 ) {
        shared_ptr<FullyConnectedLayer> layer = make_shared<FullyConnectedLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("Concat") == 0 ) {
        shared_ptr<ConcatLayer> layer = make_shared<ConcatLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("Softmax") == 0 ) {
        shared_ptr<SoftmaxLayer> layer = make_shared<SoftmaxLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("Input") == 0 ) {
        shared_ptr<InputLayer> layer = make_shared<InputLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("BatchNorm") == 0 ) {
        shared_ptr<BatchNormLayer> layer = make_shared<BatchNormLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("Scale") == 0 ) {
        shared_ptr<ScaleLayer> layer = make_shared<ScaleLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else if( type.compare("Dropout") == 0 ) {
        shared_ptr<DropoutLayer> layer = make_shared<DropoutLayer>(lparam);
        return static_pointer_cast<NNLayer>(layer);
    }
    else {
        cerr << "[ERROR] unsupported Layer type: " << type << endl;
        throw runtime_error("Network::create_layer; Not supported layer type!");
    }

}

shared_ptr<Blob> Network::get_blob_by_name(string name) {
    shared_ptr<Node> p = _name2node[name];
    auto bp = dynamic_pointer_cast<Blob>(p);
    if( bp == nullptr )
        throw runtime_error("Network::get_blob_by_name; node is not blob.");

    return bp;
}

vector<shared_ptr<NNLayer>> Network::ScheduleLayers(void) {
    map<shared_ptr<Node>, bool> vf;    // visit flag
    stack<shared_ptr<Node>> vs;        // visit stack
    vector<shared_ptr<NNLayer>> nnlayer_list;
    shared_ptr<Node> np;
    
    /* Initialize visi flags
     */
    for(uint32_t i = 0; i < _nodes.size() ; i++)
        vf.insert(make_pair(_nodes[i], false));

    /* DFS traverses
     */
    for(auto entry_nd : _entry_nodes)
        vs.push( entry_nd );

    while( ! vs.empty() ) {
        np = vs.top();
        vs.pop();

        if( vf[ np ] == false ) {
            vf[ np ] = true;

            /* Save the node pointer if the node is NNLayer
             */
            shared_ptr<NNLayer> nnlayer;
            if( (nnlayer = dynamic_pointer_cast<NNLayer>( np )) )
                nnlayer_list.push_back( nnlayer );

            for(int i = np->get_outdegree()-1; i >= 0; i--) {
                if( vf[ np->get_successor(i) ] == false ) {
                    auto succ = np->get_successor(i);
                    /* check that the node can be visited.
                     * Condition of visit: predecessors of the node
                     * should be already visited.
                     */
                    bool visit_possible = true;
                    for(auto nd : succ->get_predecessor())
                        if( vf[ nd ] == false )
                            visit_possible = false;

                    if( visit_possible )
                        vs.push( np->get_successor(i) );
                }

            }
        }
    }

    return nnlayer_list;
}



void Network::WriteNetworkToDotFile(string filename) {
    ofstream file;
    file.open( filename.c_str(), ios::out );

    file << "digraph " << _name << "{" << endl;
    file << "\trankdir = UD;" << endl;
    file << "\tnode [shape=oval]" << endl;

    vector<shared_ptr<Node>> schedList;
    schedList = BfsSchedule( _entry_nodes[0] );

    for(auto node_iter : schedList ) {
        for(auto iter : node_iter->get_successor() ) {
            if( node_iter->get_type().compare("layer") == 0 ) {
                shared_ptr<NNLayer> layer = dynamic_pointer_cast<NNLayer>(node_iter);
                file << "\t\"" << node_iter->get_name() \
                     << "\" [shape=box,style=filled,fillcolor=\".7 .3 1.0\", label=\"" \
                     << node_iter->get_name() << layer->getLayerInfoStr() << "\"]\n";
            }
            if( node_iter->get_type().compare("blob") == 0 )
                file << "\t\"" << node_iter->get_name() << "\" [fontsize=10]\n";
            file << "\t\"" << node_iter->get_name() << "\" -> \"" << iter->get_name() << "\";\n";
        }
    }

    file << "}" << endl;

    file.close();
}

}   // namespace framework
