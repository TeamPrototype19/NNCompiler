#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>
#include <string>

#include "graph.hpp"
#include "caffe.pb.h"

using namespace std;

namespace framework {

class Network : public Graph {
public:
    Network(void);
    Network(string name);
    Network(const caffe::NetParameter& net, string type);
    ~Network(void);

    void WriteNetworkToDotFile(string filename);
private:
};

}

#endif
