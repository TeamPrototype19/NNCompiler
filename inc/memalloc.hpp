#ifndef _MEMALLOC_H_
#define _MEMALLOC_H_

#include <vector>
#include <string>

#include "network.hpp"

using namespace std;

namespace framework {

typedef unsigned int Address;

typedef struct _memory_block_info {
    vector<int>  size_in_dim;
    unsigned int size_in_byte;
    int ref_counter;
    //Address address;
} memory_block_info_t;

class MemoryAlloc {
public:
    MemoryAlloc(vector<shared_ptr<NNLayer>> sched_layers, bool &success);
    ~MemoryAlloc(void);

private:
    int total_size(vector<int> size);

    /* Memory allocation algorithm ver.1
     * & its member functions.
     */
    void SetMemoryBlockInfos(vector<shared_ptr<NNLayer>> &sched_layer);
    bool MemoryAllocAlgo_v1(vector<shared_ptr<NNLayer>> &sched_layer);

    int data_unit_size;
    map<shared_ptr<Blob>, memory_block_info_t> _mblocks;
};

}

#endif
