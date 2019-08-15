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

    //shared_ptr<Blob> get_blob_by_name(string name);
    shared_ptr<NNLayer> create_layer(const caffe::LayerParameter& lparam);

    vector<shared_ptr<NNLayer>> ScheduleLayers(void);
    void WriteNetworkToDotFile(string filename);
private:
    /* Input blob dimension
     */
    vector<shared_ptr<NNLayer>>       _sched_layers;
    map<string, shared_ptr<NNLayer>>  _name2layers;
    map<string, shared_ptr<Blob>>     _name2blobs;
};

}

#endif
