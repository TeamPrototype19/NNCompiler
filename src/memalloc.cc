#include <iostream>
#include <iomanip>

#include "layer.hpp"
#include "memalloc.hpp"

using namespace std;

namespace framework {

MemoryAlloc::MemoryAlloc(vector<shared_ptr<NNLayer>> sched_layers, bool &success) {
    // TODO: check it later for reconfigurability.
    data_unit_size = sizeof(float);

    SetMemoryBlockInfos(sched_layers);
    success = MemoryAllocAlgo_v1(sched_layers);

#if 1
    /* DEBUG: Print all buffers address
     */
    cout << "---------- DEBUG: Memory allocation result ------------" << endl;
    for(auto mblk : _mblocks) {
        cout << "addr = 0x" << std::setfill('0') << std::right << std::setw(8);
        cout << std::hex << mblk.first->get_mem_addr() << "\t";
        cout << "buffer name = " << mblk.first->get_name() << "\n";
    }
#endif
}

MemoryAlloc::~MemoryAlloc(void) {
}

void MemoryAlloc::SetMemoryBlockInfos( vector<shared_ptr<NNLayer>> &sched_layer ) {

    /* Phase 1: set memory block info.
     */
    for(auto layer : sched_layer) {
        for(int i = 0 ; i < layer->GetOutBlobSize() ; i++) {
            auto blob_p = layer->GetOutBlobPtr(i);
            memory_block_info_t mblk;
            mblk.size_in_dim = blob_p->get_dim();
            mblk.size_in_byte = total_size( mblk.size_in_dim ) * data_unit_size;
            mblk.ref_counter = 0;
            //mblk.address = 0;

            _mblocks[ blob_p ] = mblk;
        }
    }

    /* Phase 2: set memory reference counter
     */
    for(auto layer : sched_layer) {
        for(int i = 0 ; i < layer->GetInBlobSize() ; i++) {
            auto blob_p = layer->GetInBlobPtr(i);

            assert( _mblocks.find( blob_p ) != _mblocks.end() );
            _mblocks[ blob_p ].ref_counter++;
        }
    }
}

bool MemoryAlloc::MemoryAllocAlgo_v1( vector<shared_ptr<NNLayer>> &sched_layer ) {
    /* simple memory allocation algorithm: 
     * + No memory use optimization
     * + Linearly allocates all output buffers of all layers
     */
    int free_address = 0;

    for(auto layer : sched_layer) {
        /* Allocates memory block for Output blob
         */
        for(int i = 0 ; i < layer->GetOutBlobSize() ; i++) {
            auto bp = layer->GetOutBlobPtr(i);

            bp->set_mem_addr( free_address );
            free_address += _mblocks[ bp ].size_in_byte;
        }
    }

    cout << "MemoryAllocAlgo_v1 result: total allocated memory = " << free_address << endl;

    return true;
}

int MemoryAlloc::total_size(vector<int> size) {
    int tsize = 1;
    for(auto s : size)
        tsize *= s;
    return tsize;
}

}   // namespace framework
