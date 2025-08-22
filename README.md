# High-Performance Text Deduplication Toolkit (Rust Edition)

This toolkit provides a highly optimized solution for large-scale text deduplication, completely rewritten in Rust for maximum performance, safety, and efficiency. It employs a multi-stage pipeline that combines exact substring deduplication using Content-Defined Chunking (CDC) with near-duplicate detection using SimHash and Faiss.

This tool is ideal for cleaning large text datasets for **training large language models**, data analysis, or any application requiring the removal of both exact and near-duplicate content. It scales almost linearly with the number of CPU cores and data size.

Project is rewrite of https://github.com/conanhujinming/text_dedup in Rust

## Features

- **Multi-Stage Deduplication Pipeline**:
  1. **Exact Substring Deduplication (CDC)**: A fast, parallelized Content-Defined Chunking stage removes redundant text blocks across the entire dataset, significantly reducing data volume.
  2. **Near-Duplicate Document Detection**: A SimHash and Faiss-powered stage identifies and removes documents that are nearly identical.
- **High-Performance Rust Core**: The core logic is written in Rust and leverages modern libraries for maximum performance:
  - **Parallel Processing**: Uses Rayon to utilize all available CPU cores safely and efficiently.
  - **Faiss Integration**: Integrates with Faiss for efficient similarity search on binary vectors.
  - **AVX2 Vectorization**: Uses SIMD instructions to accelerate SimHash signature generation on compatible CPUs.
- **Robust and Safe**: Rust's ownership model prevents common C++ memory errors, and the implementation correctly handles Unicode text.
- **Seamless Python Integration**: Exposes a clean Python API via PyO3 and is built with `maturin`, integrating smoothly with libraries like Hugging Face `datasets`.

## Requirements

### 1. System Dependencies
- **Rust Toolchain**: Install via [rustup](https://rustup.rs/).
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    Faiss: You need to have the Faiss C++ library installed. Please follow the official Faiss installation guide.
    For CPU-only on Linux:
        
    sudo apt-get install libopenblas-dev libomp-dev
    git clone https://github.com/facebookresearch/faiss.git
    cd faiss
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON
    make -C build -j$(nproc)
    sudo make -C build install

2. Python Dependencies

The project requires a few Python packages.

pip install -r requirements.txt

Installation

This project uses maturin to compile the Rust extension. Once all system and Python dependencies are met, you can install the toolkit directly using pip.
    
# Clone this repository
git clone <your-repo-url>
cd <your-repo-name>

# Install the package. This will compile the Rust core and install the Python wrapper.
pip install .

# For development, you can build in-place:
# pip install maturin
# maturin develop

The installation command will automatically invoke Cargo to build the Rust library and place it correctly within your Python environment.
Usage

The toolkit is designed to work seamlessly with Hugging Face datasets.
    
import dedup_rust_core  # This is the compiled Rust module
from datasets import Dataset

# 1. Load your dataset
dataset = Dataset.from_dict({
    'text': [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy canine.",
        "The quick brown fox jumps over the lazy dog again.",
    ]
})

# 2. Run the deduplication
# The Rust function returns a list where duplicates are replaced with `None`.
updated_docs_or_none = dedup_rust_core.deduplicate_rust(
    docs=dataset['text'],
    min_length_dedup=50,
    hamming_threshold=3,
    faiss_index_type="hash",
    simhash_bits=64
)

# 3. Integrate results back into the dataset
dataset = dataset.add_column("updated_text", updated_docs_or_none)
final_dataset = dataset.filter(lambda ex: ex["updated_text"] is not None)
final_dataset = final_dataset.remove_columns(["text"]).rename_column("updated_text", "text")

print(final_dataset.to_pandas())

How It Works

The deduplication process is a pipeline:

    Content-Defined Chunking (CDC): Each document is broken down into chunks using a rolling hash. A global set of seen chunk hashes is maintained to discard any duplicate chunk after its first appearance.

    Text Reconstruction: After discarding duplicate chunks, the remaining unique chunks for each document are concatenated to form a cleaned version of the text.

    SimHash Generation: Each cleaned document is featurized and a SimHash signature (a compact binary fingerprint) is generated in parallel.

    Faiss Indexing and Search: All SimHash signatures are added to a Faiss index for an efficient nearest-neighbor search to find all pairs within the specified hamming_threshold.

    Clustering and Filtering: A Union-Find data structure groups documents into clusters of near-duplicates. For each cluster, only one document (the one with the lowest original index) is kept.