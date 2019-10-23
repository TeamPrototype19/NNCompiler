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

enum {
    N = 0,
    C = 1,
    H = 2,
    W = 3
};

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
    bool isSameSize(vector<int> dim);

    void add_producer(shared_ptr<NNLayer> lp);
    void add_consumer(shared_ptr<NNLayer> lp);
    void set_producer(int i, shared_ptr<NNLayer> lp);
    void set_consumer(int i, shared_ptr<NNLayer> lp);
    void set_producer(vector<shared_ptr<NNLayer>> lp);
    void set_consumer(vector<shared_ptr<NNLayer>> lp);

    void set_mem_addr(unsigned long addr);
    unsigned long get_mem_addr(void);

private:
    vector<int> _dim;
    unsigned long _addr;
};

}

#endif
