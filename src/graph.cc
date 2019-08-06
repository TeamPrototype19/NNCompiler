#include <iostream>
#include <fstream>
#include <queue>
#include <stack>

#include "graph.hpp"

using namespace std;

namespace framework {

Graph::Graph(void) {
}

Graph::~Graph(void) {
    _nodes.clear();
}

void Graph::PrintInfo(void) {
    cout << "graph name: " << _name << endl;
    cout << "# of nodes: " << _nodes.size() << endl;

    for(uint32_t i = 0; i < _entry_nodes.size(); i++)
        cout << "entry node name: " << _entry_nodes[i]->get_name() << endl;

    for(uint32_t i = 0; i < _exit_nodes.size(); i++)
        cout << "exit  node name: " << _exit_nodes[i]->get_name() << endl;

    cout << "======== Node list ========" << endl;
    for(uint32_t i = 0; i < _nodes.size(); i++) {
        shared_ptr<Node> np = _nodes[i];
        cout << "  + node name   : " << np->get_name() << endl;
        cout << "  + node type   : " << np->get_type() << endl;
        cout << "  + predecessors: ";
        for(int j = 0; j < np->get_indegree(); j++)
            cout << np->get_predecessor(j)->get_name() << "  ";
        cout << endl;

        cout << "  + successors  : ";
        for(int j = 0; j < np->get_outdegree(); j++)
            cout << np->get_successor(j)->get_name() << "  ";
        cout << endl;

        cout << endl;
    }

    return;
}

void Graph::WriteGraphToDotFile(string filename) {
    ofstream file;
    file.open( filename.c_str(), ios::out );

    file << "graph " << _name << "{" << endl;
    file << "\trankdir = UD;" << endl;
    file << "node [shape=box]" << endl;

    vector<shared_ptr<Node>> schedList;
    schedList = BfsSchedule( _entry_nodes[0] );

    for(auto node_iter : schedList ) {
        for(auto iter : node_iter->get_successor() ) {
            file << "\t\"" << node_iter->get_name();
            file << "\t->\"" << iter->get_name() << "\";\n";
        }
    }

    file << "}" << endl;

    file.close();
}

vector<shared_ptr<Node>> Graph::BfsSchedule(shared_ptr<Node> np) {
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

vector<shared_ptr<Node>> Graph::DfsSchedule(shared_ptr<Node> np) {
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

}   // namespace framework
