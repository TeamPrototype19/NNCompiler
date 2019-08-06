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

    for(int i = 0; i < net.input_size(); i++) {
        string name = net.input(i)+"_node";
        shared_ptr<Layer> p_layer = make_shared<Layer>(name);
        _nodes.push_back( p_layer );
        _name2node.insert( make_pair(p_layer->get_name(), p_layer) );

        vector<int> dim;
        for(int j = 0; j < net.input_dim_size(); j++)
            dim.push_back( net.input_dim(j) );

        shared_ptr<Blob> p_blob = make_shared<Blob>(net.input(i));
        p_layer->add_successor( p_blob );
        p_blob->add_predecessor( p_layer );
        _nodes.push_back( p_blob );
        _name2node.insert( make_pair(p_blob->get_name(), p_blob) );
    }

    /* Layer processing
     */
    for(int i = 0; i < net.layer_size(); i++) {
        const caffe::LayerParameter& layer = net.layer(i);
        shared_ptr<Layer> p_layer = make_shared<Layer>(layer);
        _nodes.push_back( p_layer );
        _name2node.insert( make_pair(p_layer->get_name(), p_layer) );

        cout << "processing... : " << layer.name() << endl;

        /* Connection
         */
        for(int j = 0; j < layer.bottom_size(); j++) {
            string blob_name = layer.bottom(j);
            if( inplace_name.find(blob_name) != inplace_name.end() )
                blob_name = inplace_name[blob_name];
            shared_ptr<Node> p_blob = _name2node[blob_name];
            p_layer->add_predecessor( p_blob );
            p_blob->add_successor( p_layer );
        }

        /* Creates blobs and top connection
         */
        for(int j = 0; j < layer.top_size(); j++) {
            string blob_name = layer.top(j);

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
            p_layer->add_successor( p_blob );
            p_blob->add_predecessor( p_layer );
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
    
    return;
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
            if( node_iter->get_type().compare("layer") == 0 )
                file << "\t\"" << node_iter->get_name() << "\" [shape=box,style=filled,fillcolor=\".7 .3 1.0\"]\n";
            if( node_iter->get_type().compare("blob") == 0 )
                file << "\t\"" << node_iter->get_name() << "\" [fontsize=10]\n";
            file << "\t\"" << node_iter->get_name() << "\" -> \"" << iter->get_name() << "\";\n";
        }
    }

    file << "}" << endl;

    file.close();
}

#if 0
vector<shared_ptr<Node>> Network::BfsSchedule(shared_ptr<Node> np) {
    map<shared_ptr<Node>, bool> vf;    // visit flag
    queue<shared_ptr<Node>> vq;        // visit queue
    vector<shared_ptr<Node>> sched;    // BFS scheduled node pointers

    /* Initialize visi flags
     */
    for(uint32_t i = 0; i < _nodes.size() ; i++)
        vf.insert(make_pair(_nodes[i], false));

    /* BFS traverses
     */
    vq.push( np );
    while( ! vq.empty() ) {
        np = vq.front();
        vq.pop();

        if( vf[ np ] == false ) {
            sched.push_back( np );
            vf[ np ] = true;
            for(int i = 0; i < np->get_outdegree(); i++) {
                if( vf[ np->get_successor(i) ] == false )
                    vq.push( np->get_successor(i) );
            }
        }
    }

    return sched;
}

vector<shared_ptr<Node>> Network::DfsSchedule(shared_ptr<Node> np) {
    map<shared_ptr<Node>, bool> vf;    // visit flag
    stack<shared_ptr<Node>> vs;        // visit stack
    vector<shared_ptr<Node>> sched;    // BFS scheduled node pointers

    /* Initialize visi flags
     */
    for(uint32_t i = 0; i < _nodes.size() ; i++)
        vf.insert(make_pair(_nodes[i], false));

    /* BFS traverses
     */
    vs.push( np );
    while( ! vs.empty() ) {
        np = vs.top();
        vs.pop();

        if( vf[ np ] == false ) {
            sched.push_back( np );
            vf[ np ] = true;
            for(int i = np->get_outdegree()-1; i >= 0; i--) {
                if( vf[ np->get_successor(i) ] == false )
                    vs.push( np->get_successor(i) );
            }
        }
    }

    return sched;
}
#endif

}   // namespace framework
