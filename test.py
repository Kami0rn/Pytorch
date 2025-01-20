import torch
import time

# Define a simple matrix multiplication
def benchmark(device):
    print(f"Running on {device}")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    # Start the timer
    start_time = time.time()
    
    # Perform matrix multiplication
    for _ in range(100):
        c = torch.matmul(a, b)
    
    # Stop the timer
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return elapsed_time

# Benchmark CPU
benchmark("cpu")

# If CUDA is available, benchmark GPU
if torch.cuda.is_available():
    benchmark("cuda")
else:
    print("GPU not available on this system.")
