use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

mod cdc;
mod simhash;
mod utils;

use crate::cdc::get_chunks_and_hashes;
use crate::simhash::get_document_simhash;
use crate::utils::{clean_utf8_bytes, UnionFind};
use faiss::{index::IndexBinary, Idx};

#[pyfunction]
#[pyo3(name = "deduplicate_rust")]
fn deduplicate_py(
    docs: Vec<String>,
    min_length_dedup: usize,
    hamming_threshold: u32,
    faiss_index_type: String,
    simhash_bits: usize,
) -> PyResult<Vec<Option<String>>> {
    // Initialize logging to see diagnostic output
    let _ = env_logger::builder().filter_level(log::LevelFilter::Info).try_init();

    // --- Stage 1: Parallel CDC Deduplication ---
    log::info!("--- Stage 1: Rust Parallel CDC Deduplication ---");
    let stage1_start = Instant::now();

    let doc_chunks: Vec<Vec<&[u8]>> = docs
        .par_iter()
        .map(|doc| {
            let (chunks, _) = get_chunks_and_hashes(doc.as_bytes(), min_length_dedup, 16);
            chunks
        })
        .collect();

    let mut global_seen_hashes = HashSet::new();
    let chunks_to_keep_per_doc: Vec<Vec<&[u8]>> = doc_chunks
        .into_iter()
        .map(|chunks| {
            chunks
                .into_iter()
                .filter(|chunk| {
                    let hash = xxhash_rust::xxh3::xxh3_64(chunk);
                    global_seen_hashes.insert(hash)
                })
                .collect()
        })
        .collect();

    let deduped_texts: Vec<String> = chunks_to_keep_per_doc
        .par_iter()
        .map(|chunks| String::from_utf8_lossy(&chunks.concat()).into_owned())
        .collect();

    let stage1_duration = stage1_start.elapsed();

    // --- Diagnostics for Stage 1 ---
    let original_size_bytes: usize = docs.iter().map(|s| s.len()).sum();
    let deduped_size_bytes: usize = deduped_texts.iter().map(|s| s.len()).sum();
    let reduction_ratio = if original_size_bytes > 0 {
        100.0 * (1.0 - (deduped_size_bytes as f64 / original_size_bytes as f64))
    } else {
        0.0
    };
    log::info!("--- Stage 1 Diagnostics (took {:.2?}s) ---", stage1_duration);
    log::info!("Original data size: {:.2} MB", original_size_bytes as f64 / (1024.0 * 1024.0));
    log::info!("Data size after CDC: {:.2} MB", deduped_size_bytes as f64 / (1024.0 * 1024.0));
    log::info!("CDC reduction ratio: {:.2}%", reduction_ratio);

    // --- Stage 2: Parallel SimHash Signature Generation ---
    log::info!("--- Stage 2: Rust Parallel SimHash Generation ---");
    let stage2_start = Instant::now();
    
    let valid_signatures: Vec<(usize, Vec<u64>)> = deduped_texts
        .par_iter()
        .enumerate()
        .filter_map(|(i, text)| {
            if !text.is_empty() {
                Some((i, get_document_simhash(text.as_bytes(), simhash_bits)))
            } else {
                None
            }
        })
        .collect();
        
    let stage2_duration = stage2_start.elapsed();
    log::info!("SimHash generation took {:.2?}s", stage2_duration);


    // --- Stage 3: Faiss Near-Duplicate Detection ---
    log::info!("--- Stage 3: Rust Faiss Near-Duplicate Detection ---");
    let stage3_start = Instant::now();
    
    if valid_signatures.is_empty() {
        return Ok(vec![]);
    }

    let num_valid_docs = valid_signatures.len();
    let hash_bytes = simhash_bits / 8;

    let mut binary_vectors = Vec::with_capacity(num_valid_docs * hash_bytes);
    for (_, sig) in &valid_signatures {
        for &val in sig {
            binary_vectors.extend_from_slice(&val.to_le_bytes());
        }
    }

    let mut index = match faiss_index_type.as_str() {
        "flat" => faiss::index::index_binary_factory(simhash_bits as i32, "BFlat")?,
        "hash" => faiss::index::index_binary_factory(simhash_bits as i32, "BHash64")?,
        "HNSW" => faiss::index::index_binary_factory(simhash_bits as i32, "BHNSW32")?,
        "IVF" => {
             let nlist = if num_valid_docs < 10000 {
                std::cmp::max(1, num_valid_docs / 16)
            } else {
                (4.0 * (num_valid_docs as f64).sqrt()) as usize
            };
            let nlist = std::cmp::min(nlist, 65536);
            let nlist = std::cmp::max(nlist, 1);
            let factory_str = format!("BIVF{},BFlat", nlist);
            faiss::index::index_binary_factory(simhash_bits as i32, &factory_str)?
        }
        _ => {
            log::warn!("Unknown Faiss index type '{}'. Falling back to 'flat'.", faiss_index_type);
            faiss::index::index_binary_factory(simhash_bits as i32, "BFlat")?
        }
    };

    if !index.is_trained() {
        log::info!("Training Faiss {} index...", faiss_index_type);
        let train_size = std::cmp::min(num_valid_docs, 256 * 100);
        if train_size > 0 {
            // faiss-rs wants a slice of the original data for training.
            let training_data = &binary_vectors[..train_size * hash_bytes];
            index.train(training_data)?;
        }
    }

    log::info!("Adding {} vectors to Faiss index...", num_valid_docs);
    index.add(&binary_vectors)?;

    let res = index.range_search(&binary_vectors, hamming_threshold as f32)?;

    // --- Find connected components using Union-Find ---
    let mut uf = UnionFind::new(num_valid_docs);
    for (i, neighbors) in res.iter().enumerate() {
        for neighbor in neighbors {
            uf.unite(i, neighbor.label as usize);
        }
    }
    
    let mut components: hashbrown::HashMap<usize, Vec<usize>> = hashbrown::HashMap::new();
    for i in 0..num_valid_docs {
        let root = uf.find(i);
        components.entry(root).or_default().push(i);
    }

    let mut to_remove = HashSet::new();
    for (_, component_indices) in components {
        if component_indices.len() > 1 {
            let min_original_index = component_indices.iter().map(|&local_idx| valid_signatures[local_idx].0).min().unwrap();
            for &local_idx in &component_indices {
                let original_idx = valid_signatures[local_idx].0;
                if original_idx != min_original_index {
                    to_remove.insert(original_idx);
                }
            }
        }
    }

    let mut final_results: Vec<Option<String>> = deduped_texts
        .into_iter()
        .map(|text| Some(String::from_utf8_lossy(&clean_utf8_bytes(text.as_bytes())).into_owned()))
        .collect();
    
    for idx in to_remove {
        if idx < final_results.len() {
            final_results[idx] = None;
        }
    }

    let stage3_duration = stage3_start.elapsed();
    log::info!("Faiss near-duplicate detection took {:.2?}s", stage3_duration);

    Ok(final_results)
}

#[pymodule]
fn dedup_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deduplicate_py, m)?)?;
    Ok(())
}