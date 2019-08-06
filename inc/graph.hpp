#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "node.hpp"

using namespace std;

namespace framework {

class Graph {
public:
    Graph(void);
    ~Graph(void);

    void PrintInfo(void);
    void WriteGraphToDotFile(string filename);
    vector<shared_ptr<Node>> BfsSchedule(shared_ptr<Node> np);
    vector<shared_ptr<Node>> DfsSchedule(shared_ptr<Node> np);

protected:
    string _name;
    string _type;
    vector<shared_ptr<Node>>         _nodes;
    map<string, shared_ptr<Node>>    _name2node;
    vector<shared_ptr<Node>>         _entry_nodes;
    vector<shared_ptr<Node>>         _exit_nodes;
};

}

#endif
