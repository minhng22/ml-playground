import gc

def run_garbage_collector():
    print("Running garbage collector...")
    gc.collect()
    print("Garbage collection complete.")

if __name__ == "__main__":
    run_garbage_collector()