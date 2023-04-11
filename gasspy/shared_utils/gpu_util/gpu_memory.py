import cupy


def free_memory():
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()