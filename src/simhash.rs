use hashbrown::HashMap;
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Generates a SimHash fingerprint for a document.
/// This version is highly optimized, using AVX2 intrinsics if available.
pub fn get_document_simhash(text: &[u8], hashbits: usize) -> Vec<u64> {
    let num_blocks = (hashbits + 63) / 64;
    
    // Stage 1: Fast, zero-copy feature counting
    let mut features = HashMap::new();
    for word in text.split(|c| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()) {
        // This is not perfectly zero-copy as it allocates for lowercase, but it's efficient.
        let lower_word = word.to_ascii_lowercase();
        *features.entry(lower_word).or_insert(0) += 1;
    }

    if features.is_empty() {
        return vec![0; num_blocks];
    }
    
    let mut v = vec![0i32; hashbits];

    // Stage 2: Calculate the weighted vector 'v'
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            #[cfg(target_arch = "x86_64")]
            unsafe { avx2_path(&features, &mut v, hashbits) };
        } else {
            scalar_path(&features, &mut v, hashbits);
        }
    }
    
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        scalar_path(&features, &mut v, hashbits);
    }

    // Stage 3: Finalize the fingerprint
    let mut fingerprint = vec![0u64; num_blocks];
    for i in 0..hashbits {
        if v[i] >= 0 {
            fingerprint[i / 64] |= 1u64 << (i % 64);
        }
    }
    fingerprint
}

#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn avx2_path(features: &HashMap<Vec<u8>, i32>, v: &mut [i32], hashbits: usize) {
    for (feature, &weight) in features {
        let feature_hash = xxh3_64(feature);
        let weights_add = _mm256_set1_epi32(weight);
        let weights_sub = _mm256_set1_epi32(-weight);

        let mut i = 0;
        while i <= hashbits.saturating_sub(8) {
            // Pre-compute all hash values to reduce xxh3 call overhead
            let hash_0 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), i as u64) as i32;
            let hash_1 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 1) as u64) as i32;
            let hash_2 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 2) as u64) as i32;
            let hash_3 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 3) as u64) as i32;
            let hash_4 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 4) as u64) as i32;
            let hash_5 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 5) as u64) as i32;
            let hash_6 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 6) as u64) as i32;
            let hash_7 = xxh3_64_with_seed(&feature_hash.to_le_bytes(), (i + 7) as u64) as i32;
            
            let prng_hash = _mm256_set_epi32(
                hash_7, hash_6, hash_5, hash_4, hash_3, hash_2, hash_1, hash_0
            );

            let v_current = _mm256_loadu_si256(v.as_ptr().add(i) as *const __m256i);
            
            // Create a mask where lanes are all 1s for positive hashes, all 0s for negative.
            let mask = _mm256_srai_epi32(prng_hash, 31);
            
            // Blend using the mask to choose between adding or subtracting weight.
            let delta = _mm256_blendv_epi8(weights_sub, weights_add, mask);
            
            let v_new = _mm256_add_epi32(v_current, delta);
            _mm256_storeu_si256(v.as_mut_ptr().add(i) as *mut __m256i, v_new);

            i += 8;
        }

        // Handle remainder
        for i in i..hashbits {
            let h = xxh3_64_with_seed(&feature_hash.to_le_bytes(), i as u64);
            if (h & 1) == 1 {
                v[i] += weight;
            } else {
                v[i] -= weight;
            }
        }
    }
}

fn scalar_path(features: &HashMap<Vec<u8>, i32>, v: &mut [i32], hashbits: usize) {
    for (feature, &weight) in features {
        let feature_hash = xxh3_64(feature);
        for i in 0..hashbits {
            let h = xxh3_64_with_seed(&feature_hash.to_le_bytes(), i as u64);
            if (h & 1) == 1 {
                v[i] += weight;
            } else {
                v[i] -= weight;
            }
        }
    }
}