import time
import numpy as np
import sys
import gc

def run_memory_benchmark(size_mb=4000, iterations=5):
    """
    Measures raw host memory bandwidth using numpy array copy operations.
    Attempts to saturate dual-channel DDR5 by reading/writing a large block of memory.
    """
    print("============================================================================")
    print("  PHANTOM MEMORY BANDWIDTH DIAGNOSTIC")
    print("============================================================================")
    print(f"Allocating {size_mb} MB array for testing...")
    
    # Use float64 for 8-byte alignment, typical of high-bandwidth ML operations
    num_elements = (size_mb * 1024 * 1024) // 8
    
    try:
        # Allocate source array
        src = np.ones(num_elements, dtype=np.float64)
        # Allocate destination array
        dst = np.empty_like(src)
    except MemoryError:
        print(f"[ERROR] Failed to allocate {size_mb * 2} MB. Try a smaller size.")
        return
        
    print(f"Testing Sequential Read/Write Bandwidth ({iterations} iterations)...")
    
    bandwidths = []
    
    for i in range(iterations):
        # Force garbage collection to minimize noise
        gc.collect()
        
        start_time = time.perf_counter()
        # np.copyto represents a streaming read of src and write of dst
        np.copyto(dst, src)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        # Size in GB = size_mb / 1024. 
        # Since we read size_mb and write size_mb, total traffic is 2 * size_mb
        traffic_gb = (size_mb * 2) / 1024.0
        bw_gbs = traffic_gb / elapsed
        bandwidths.append(bw_gbs)
        
        print(f"  Iteration {i+1}: {elapsed:.4f} s | {bw_gbs:.2f} GB/s")
        
    avg_bw = sum(bandwidths) / len(bandwidths)
    peak_bw = max(bandwidths)
    
    print("\n----------------------------------------------------------------------------")
    print("  RESULTS")
    print("----------------------------------------------------------------------------")
    print(f"  Average Bandwidth: {avg_bw:.2f} GB/s")
    print(f"  Peak Bandwidth:    {peak_bw:.2f} GB/s")
    
    print("\n[DIAGNOSTIC]")
    if avg_bw < 30.0:
        print("=> WARNING: Measured bandwidth is well below typical DDR5 dual-channel limits.")
        print("   Your 24GB configuration may be running in ASYMMETRIC SINGLE-CHANNEL mode.")
        print("   This severely throttles Dense 32B generation speeds.")
    elif avg_bw < 45.0:
        print("=> OK: Bandwidth is consistent with lower-end or asymmetric dual-channel DDR5.")
    else:
        print("=> EXCELLENT: Bandwidth saturates dual-channel DDR5 expectations (~48+ GB/s).")
    print("============================================================================")

if __name__ == "__main__":
    # Test using 4GB array (reads 4GB, writes 4GB = 8GB total traffic)
    run_memory_benchmark(size_mb=4000, iterations=5)
