use std::cmp::max;
use xxhash_rust::xxh3::xxh3_64;

struct RollingHash {
    base: u64,
    prime: u64,
    hash: u64,
    power: u64,
}

impl RollingHash {
    fn new(base: u64, prime: u64) -> Self {
        RollingHash {
            base,
            prime,
            hash: 0,
            power: 1,
        }
    }

    fn generate(&mut self, window: &[u8]) {
        let window_size = window.len();
        self.power = 1;
        for _ in 0..window_size.saturating_sub(1) {
            self.power = (self.power * self.base) % self.prime;
        }

        self.hash = 0;
        for &byte in window {
            self.hash = (self.hash * self.base + byte as u64) % self.prime;
        }
    }

    fn slide(&mut self, old_byte: u8, new_byte: u8) {
        let old_val = (old_byte as u64 * self.power) % self.prime;
        self.hash = (self.hash + self.prime - old_val) % self.prime;
        self.hash = (self.hash * self.base + new_byte as u64) % self.prime;
    }
}

pub fn get_chunks_and_hashes<'a>(
    text: &'a [u8],
    min_length_dedup: usize,
    window_size: usize,
) -> (Vec<&'a [u8]>, Vec<u64>) {
    let divisor = max(1, min_length_dedup) as u64;

    if text.len() < min_length_dedup {
        return if text.is_empty() {
            (vec![], vec![])
        } else {
            (vec![text], vec![xxh3_64(text)])
        };
    }

    let mut chunks = Vec::new();
    let mut start_pos = 0;

    if text.len() <= window_size {
        chunks.push(text);
    } else {
        let mut rh = RollingHash::new(257, 1_000_000_007);
        rh.generate(&text[..window_size]);

        for i in 0..=text.len() - window_size {
            let current_pos_end_of_window = i + window_size;
            let current_chunk_length = current_pos_end_of_window - start_pos;

            if current_chunk_length >= min_length_dedup {
                if (rh.hash % divisor) == 0 {
                    chunks.push(&text[start_pos..current_pos_end_of_window]);
                    start_pos = current_pos_end_of_window;
                }
            }

            if i < text.len() - window_size {
                rh.slide(text[i], text[i + window_size]);
            }
        }

        if start_pos < text.len() {
            chunks.push(&text[start_pos..]);
        }
    }
    
    let hashes = chunks.iter().map(|c| xxh3_64(c)).collect();
    (chunks, hashes)
}