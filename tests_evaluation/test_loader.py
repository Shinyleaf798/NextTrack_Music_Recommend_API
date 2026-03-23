import time
import sys
import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

def test_performance():
    print("=== NextTrack Memory Index Performance Test ===")
    
    start_load = time.time()
    from NextTrack_back.utils.data_loader import data_store
    end_load = time.time()
    load_duration = end_load - start_load

    if data_store.df is not None:
        total_tracks = len(data_store.df)
        print(f"Total Tracks Loaded: {total_tracks}")
        print(f"Initial Loading Time (CSV to RAM): {load_duration:.4f} seconds")
        
        test_id = data_store.df['track_id'].iloc[0]
        
        start_search = time.time()
        track_ram = data_store.get_track_by_id(test_id)
        end_search = time.time()
        
        search_latency_ms = (end_search - start_search) * 1000
        print(f"\n--- [Test A] In-Memory Search ---")
        print(f"RAM Search Latency: {search_latency_ms:.4f} ms")
        
        print(f"\n--- [Test B] Disk I/O Simulation ---")
        
        start_disk = time.time()
        temp_df = pd.read_csv(data_store.file_path)
        track_disk = temp_df[temp_df['track_id'] == test_id].iloc[0]
        end_disk = time.time()
        
        disk_latency_ms = (end_disk - start_disk) * 1000
        print(f"Disk I/O Latency: {disk_latency_ms:.4f} ms")
        
        speedup = disk_latency_ms / search_latency_ms if search_latency_ms > 0 else 0
        print(f"\nRAM search speedup: {speedup:.2f}x")
        
        if search_latency_ms < 100:
            print("Status: SUCCESS (<100ms)")
        else:
            print("Status: FAILED")
            
    else:
        print("Status: FAILED - Dataset not found or empty.")

if __name__ == "__main__":
    test_performance()