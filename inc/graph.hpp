#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>
#include <string>

#include "node.hpp"

using namespace std;

namespace framework {

class Graph {
public:
    Graph(void);
    ~Graph(void);

    void PrintInfo(void);
    void WriteGraph2DotFile(string filename);
    vector<shared_ptr<Node>> BfsSchedule(shared_ptr<Node> np);
    vector<shared_ptr<Node>> DfsSchedule(shared_ptr<Node> np);

protected:
    string _name;
    string _type;
    map<string, shared_ptr<Node>>    _nodes;
    vector<shared_ptr<Node>>         _entry_nodes;
    vector<shared_ptr<Node>>         _exit_nodes;
};

}

#endif
