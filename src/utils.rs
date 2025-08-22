use petgraph::unionfind::UnionFind as PetgraphUnionFind;

pub struct UnionFind {
    uf: PetgraphUnionFind<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            uf: PetgraphUnionFind::new(n),
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        self.uf.find(i)
    }

    pub fn unite(&mut self, i: usize, j: usize) {
        self.uf.union(i, j);
    }
}

/// Removes invalid UTF-8 byte sequences from a byte slice.
/// This function mimics the behavior of the original C++ `clean_utf8`,
/// which skips invalid bytes rather than replacing them.
pub fn clean_utf8_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte < 0x80 { // ASCII
            output.push(byte);
            i += 1;
            continue;
        }
        
        // Determine expected length of a multi-byte sequence
        let len = if (byte & 0xE0) == 0xC0 { 2 }      // 2-byte sequence
                  else if (byte & 0xF0) == 0xE0 { 3 } // 3-byte sequence
                  else if (byte & 0xF8) == 0xF0 { 4 } // 4-byte sequence
                  else {
                      // Invalid start byte, skip it
                      i += 1;
                      continue;
                  };

        // Check if the sequence is complete within the input slice
        if i + len > bytes.len() {
            // Incomplete sequence at the end, drop it
            break;
        }

        // Check if all continuation bytes are valid (start with 10xxxxxx)
        let is_valid = (1..len).all(|j| (bytes[i + j] & 0xC0) == 0x80);
        
        if is_valid {
            // Append the valid sequence
            output.extend_from_slice(&bytes[i..i + len]);
            i += len;
        } else {
            // Invalid sequence, skip only the start byte and re-evaluate
            i += 1;
        }
    }
    output
}