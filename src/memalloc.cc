#include <iostream>
#include <iomanip>

#include "layer.hpp"
#include "memalloc.hpp"

using namespace std;

namespace framework {

MemoryAlloc::MemoryAlloc(CompileContext &context) {
    // TODO: check it later for reconfigurability.
    data_unit_size = sizeof(float);

    SetMemoryBlockInfos(context);
    context.compile_result = MemoryAllocAlgo_v1(context);

#if 1
    /* DEBUG: Print all buffers address
     */
    logfs << "---------- DEBUG: Memory allocation report ------------" << endl;
    for(auto mblk : _mblocks) {
        logfs << "buffer name = " << std::right << std::setw(30) << mblk.first->get_name() << "     ";
        logfs << "addr = 0x" << std::setfill('0') << std::right << std::setw(8) \
              << std::hex << mblk.first->get_mem_addr() << std::dec << std::setfill(' ') << "     ";
        logfs << "size = " << std::right << std::setw(8) << mblk.second.size_in_byte << "(Bytes)" \
              << " (" << (mblk.second.size_in_byte/(1000)) << " KB)\n";
    }
    logfs << "-------------------------------------------------------" << endl;
#endif
}

MemoryAlloc::~MemoryAlloc(void) {
}

void MemoryAlloc::SetMemoryBlockInfos( CompileContext &context ) {

    /* Phase 1: set memory block info.
     */
    // check that the entry node is blob or layer.
    // if entry node is blob, then add it into _mblocks.
    for(auto node : *(context._entry_nodes)) {
        shared_ptr<Blob> blob_p = dynamic_pointer_cast<Blob>(node);
        if( blob_p != nullptr ) {
            memory_block_info_t mblk;
            mblk.size_in_dim = blob_p->get_dim();
            mblk.size_in_byte = total_size( mblk.size_in_dim ) * data_unit_size;
            mblk.ref_counter = 0;

            _mblocks[ blob_p ] = mblk;
        }
    }

    // add output blobs of each layer nodes.
    for(auto layer : *(context._sched_layers)) {
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
    for(auto layer : *(context._sched_layers) ) {
        for(int i = 0 ; i < layer->GetInBlobSize() ; i++) {
            auto blob_p = layer->GetInBlobPtr(i);

            assert( _mblocks.find( blob_p ) != _mblocks.end() );
            _mblocks[ blob_p ].ref_counter++;
        }
    }
}

bool MemoryAlloc::MemoryAllocAlgo_v1( CompileContext &context ) {
    /* simple memory allocation algorithm: 
     * + No memory use optimization
     * + Linearly allocates all output buffers of all layers
     */
    unsigned long free_address = 0;

    /* Memory allocation for entry_nodes (not layer type) firstly.
     */
    for(auto node : *(context._entry_nodes) )  {
        auto bp = dynamic_pointer_cast<Blob>(node);
        if( bp ) {
            bp->set_mem_addr( free_address );
            free_address += (unsigned long) _mblocks[ bp ].size_in_byte;
        }
    }

    for(auto layer : *(context._sched_layers) )  {
        /* Allocates memory block for Output blob
         */
        for(int i = 0 ; i < layer->GetOutBlobSize() ; i++) {
            auto bp = layer->GetOutBlobPtr(i);

            bp->set_mem_addr( free_address );
            free_address += (unsigned long) _mblocks[ bp ].size_in_byte;
        }
    }

    context.total_buffer_size = free_address;

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
