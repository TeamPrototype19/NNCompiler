#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>
#include <string>

#include "types.h"
#include "blob.hpp"
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

    shared_ptr<Blob> get_blob_by_name(string name);

    void WriteNetworkToDotFile(string filename);
private:
    /* Input blob dimension
     */
    Blob    input_blob;
};

}

#endif
