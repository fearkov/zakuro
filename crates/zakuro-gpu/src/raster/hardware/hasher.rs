//! a quick hash for the maps keyed by what the emulator makes itself,
//! pipelines, programs, their translations and samplers. the standard one,
//! SipHash, guards against keys picked to collide, which these never are,
//! and every draw looks a few of them up, a pipeline's key being some 150
//! bytes.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

pub(super) type QuickMap<K, V> = HashMap<K, V, BuildHasherDefault<Quick>>;
pub(super) type QuickSet<K> = HashSet<K, BuildHasherDefault<Quick>>;

/// odd, with its bits spread over the word, rustc's own hasher uses it.
const K: u64 = 0xF135_7AEA_2E62_A9C5;

/// adds in a word at a time and multiplies, the way rustc's FxHasher does.
#[derive(Default, Clone, Copy)]
pub(super) struct Quick {
    hash: u64,
}

impl Quick {
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for Quick {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        if !rest.is_empty() {
            let mut last = [0; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.add(value as u64);
    }

    fn write_u16(&mut self, value: u16) {
        self.add(value as u64);
    }

    fn write_u32(&mut self, value: u32) {
        self.add(value as u64);
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        // the multiplies leave the best mixed bits at the top, and the map
        // picks a bucket by the bottom ones, folding the halves of a wide
        // multiply together spreads every bit over both
        let wide = self.hash as u128 * K as u128;
        wide as u64 ^ (wide >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn hash(value: impl Hash) -> u64 {
        BuildHasherDefault::<Quick>::default().hash_one(value)
    }

    /// keys that differ anywhere, in a word or in the bytes after the last
    /// whole one, hash apart, and the same key the same.
    #[test]
    fn keys_that_differ_hash_apart() {
        let key = (Some(0x1234u32), 7u8, [0x0FFF_0FFFu32; 30]);
        assert_eq!(hash(key), hash(key));
        for word in 0..30 {
            for bit in [0, 13, 31] {
                let mut other = key;
                other.2[word] ^= 1 << bit;
                assert_ne!(hash(key), hash(other), "word {word} bit {bit}");
            }
        }
        assert_ne!(hash(key), hash((None::<u32>, 7u8, key.2)));
        assert_ne!(hash([1u8, 2, 3]), hash([1u8, 2, 4]));
        assert_ne!(hash(&[0u8; 9][..]), hash(&[0u8, 0, 0, 0, 0, 0, 0, 0, 1][..]));
    }

    /// addresses a page apart, as large allocations are, spread over the
    /// low bits a map picks its buckets by about as keys picked at random
    /// would, some 160 of 256.
    #[test]
    fn aligned_keys_spread_over_the_buckets() {
        let buckets: QuickSet<u64> = (0..256usize).map(|i| hash(0x7F00_0000_0000usize + i * 4096) & 255).collect();
        assert!(buckets.len() > 150, "{} buckets of 256", buckets.len());
    }
}
