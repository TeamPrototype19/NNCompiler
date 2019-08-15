#ifndef _NODE_H_
#define _NODE_H_

#include "log.h"
#include <vector>
#include <string>
#include <memory>

using namespace std;

namespace framework {

class Node {
public:
    Node(void);
    Node(string name, string type);
    virtual ~Node() = default;

    string get_name(void);
    string get_type(void);
    int get_indegree(void);
    int get_outdegree(void);
    vector<shared_ptr<Node>> get_predecessor(void);
    vector<shared_ptr<Node>> get_successor(void);
    shared_ptr<Node> get_predecessor(int i);
    shared_ptr<Node> get_successor(int i);

protected:
    string _name;
    string _type;
    vector<shared_ptr<Node>>  _successor;
    vector<shared_ptr<Node>>  _predecessor;

    void add_predecessor(shared_ptr<Node> np);
    void add_successor(shared_ptr<Node> np);
    void set_predecessor(vector<shared_ptr<Node>> np_list);
    void set_successor(vector<shared_ptr<Node>> np_list);
};

}

#endif
