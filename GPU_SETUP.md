# GPU Support Setup for Linux

This document provides instructions for compiling the text deduplication tool with GPU support on Linux systems.

## Prerequisites

### 1. CUDA Installation (for GPU support)

**Ubuntu/Debian:**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**CentOS/RHEL:**
```bash
# Install CUDA toolkit
sudo yum install -y cuda
```

### 2. Build Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libomp-dev pkg-config

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake openblas-devel libgomp
```

## Faiss Installation with GPU Support

### Option 1: Install from Conda (Recommended)

```bash
# Install miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment and install faiss-gpu
conda create -n text_dedup python=3.9
conda activate text_dedup
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
```

### Option 2: Build from Source

```bash
# Clone Faiss repository
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Configure with GPU support
cmake -B build . \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"

# Build and install
make -C build -j$(nproc)
sudo make -C build install

# Update library path
echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/faiss.conf
sudo ldconfig
```

## Building the Text Deduplication Tool

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Install Python Dependencies

```bash
pip install maturin
pip install -r requirements.txt
```

### 3. Build with GPU Support

The application now uses feature flags for conditional compilation:

**For Linux with GPU support:**
```bash
# Build with GPU features enabled
maturin develop --release --features gpu

# Or for production build
pip install . --config-settings="--features=gpu"
```

**For CPU-only builds (any platform):**
```bash
# Default build (no GPU features)
maturin develop --release

# Or for production build
pip install .
```

## Configuration

### Environment Variables

Set these environment variables for optimal GPU performance:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export OMP_NUM_THREADS=8       # Adjust based on CPU cores
export FAISS_ENABLE_GPU=1      # Enable GPU in Faiss
```

### Runtime Verification

Test that GPU support is working:

```python
import dedup_rust_core
import faiss

# Check if GPU is available
print(f"GPU count: {faiss.get_num_gpus()}")

# Test the deduplication function
docs = ["Hello world", "Hello world", "Different text"]
result = dedup_rust_core.deduplicate_rust(
    docs, 
    min_length_dedup=10, 
    hamming_threshold=5, 
    faiss_index_type="flat",  # or "IVF" for large datasets
    simhash_bits=128
)
print(f"Deduplicated: {result}")
```

## Platform-Specific Behavior

- **With `gpu` feature enabled**: Full GPU support with CUDA acceleration via Faiss
- **Without `gpu` feature (default)**: CPU-only fallback using simple hamming distance calculation
- **Recommended**: Use GPU features on Linux with CUDA, CPU-only on other platforms

## Performance Tuning

### Faiss Index Types

- **"flat"**: Exact search, best for < 1M documents
- **"IVF"**: Inverted file index, good for large datasets
- **"HNSW"**: Hierarchical navigable small world, fast approximate search
- **"hash"**: LSH-based hashing, memory efficient

### GPU Memory Management

```bash
# Monitor GPU memory usage
nvidia-smi

# Limit GPU memory if needed
export CUDA_MEM_LIMIT=8000  # Limit to 8GB
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA is in PATH and LD_LIBRARY_PATH
2. **Faiss not found**: Check that libfaiss.so is in /usr/local/lib or similar
3. **GPU out of memory**: Reduce batch size or use CPU fallback

### Debug Build

For debugging compilation issues:

```bash
RUST_LOG=debug maturin develop
```

### Fallback to CPU

If GPU setup fails, the system will automatically fall back to CPU-only mode with simple hamming distance calculation.