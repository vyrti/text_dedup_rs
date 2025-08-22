import argparse
import time
import dedup_rust_core  # The new rust module
from datasets import load_dataset
from tqdm import tqdm

def run_deduplication_with_rust_core(dataset, text_column, min_length_dedup, hamming_threshold, faiss_index_type, simhash_bits):
    """
    A wrapper function to call the Rust core for the entire deduplication process.
    """
    print("\n--- Running Deduplication with Optimized Rust Core ---")
    
    # Extract the text column into a list of strings
    docs = [doc[text_column] or "" for doc in tqdm(dataset, desc="Preparing data for Rust core")]
    
    print("Calling Rust deduplication function...")
    # The function name in Rust is `deduplicate_rust`
    dedup_result = dedup_rust_core.deduplicate_rust(
        docs,
        min_length_dedup,
        hamming_threshold,
        faiss_index_type,
        simhash_bits
    )

    # Stats are now printed from the Rust side via logging.
    return dedup_result

def main():
    """
    Main function to run the script as a standalone, centralized deduplication tool.
    """
    parser = argparse.ArgumentParser(description="A comprehensive, centralized tool for text dataset deduplication, powered by Rust.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Parquet dataset file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the deduplicated Parquet file.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text data.")
    
    parser.add_argument("--min_length_dedup", type=int, default=50, help="Minimum length of a substring for exact deduplication (CDC).")
    parser.add_argument("--faiss_index_type", type=str, choices=['hash', 'HNSW', 'IVF', 'flat'], default='flat', help="Type of Faiss index to use.")
    parser.add_argument("--hamming_threshold", type=int, default=3, help="Hamming distance threshold for near-duplicates (SimHash).")
    parser.add_argument("--simhash_bits", type=int, default=64, help="Number of bits for SimHash. Must be a multiple of 8.")

    args = parser.parse_args()
    
    if args.simhash_bits % 8 != 0:
        parser.error("--simhash_bits must be a multiple of 8.")

    print(f"Loading dataset from: {args.input_file}")
    dataset = load_dataset('parquet', data_files={'train': args.input_file})['train']
    print(f"Initial dataset size: {len(dataset):,} records")
    
    time_start = time.time()
    
    updated_docs_or_none = run_deduplication_with_rust_core(
        dataset, 
        args.text_column,
        args.min_length_dedup,
        args.hamming_threshold,
        args.faiss_index_type,
        args.simhash_bits
    )
    
    dataset = dataset.add_column("updated_text", updated_docs_or_none)
    final_dataset = dataset.filter(lambda example: example["updated_text"] is not None)

    # Clean up the columns
    final_dataset = final_dataset.remove_columns([args.text_column])
    final_dataset = final_dataset.rename_column("updated_text", args.text_column)

    time_end = time.time()
    print(f"\n--- Deduplication Summary ---")
    print(f"Total time taken: {time_end - time_start:.2f} seconds")
    print(f"Initial record count: {len(dataset):,}")
    print(f"Final record count: {len(final_dataset):,}")
    print(f"Saving final dataset to: {args.output_file}")
    final_dataset.to_parquet(args.output_file)
    print("Deduplication process completed successfully.")

if __name__ == "__main__":
    main()