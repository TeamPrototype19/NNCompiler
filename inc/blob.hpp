#ifndef _BLOB_H_
#define _BLOB_H_

#include <vector>
#include <string>
#include <memory>

#include "log.h"
#include "node.hpp"

using namespace std;

namespace framework {

class NNLayer;

class Blob : public Node {
public:
    Blob(void);
    Blob(string name);
    ~Blob(void);

    template<typename First, typename... Args>
    void set_dim(First first, Args... args);
    void set_dim(int arg);
    void set_dim(std::vector<int> dim);
    vector<int> get_dim(void);
    string getSizeInfoStr(void);

    void add_producer(shared_ptr<NNLayer> lp);
    void add_consumer(shared_ptr<NNLayer> lp);
    void set_mem_addr(unsigned int addr);
    void set_index(int index);

private:
    vector<int> _dim;
    unsigned int _addr;
    int _index;
};

}

#endif
