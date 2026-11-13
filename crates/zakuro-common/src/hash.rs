//! maps keyed by small integers, handles and page numbers, hashed by a
//! multiply rather than SipHash, as the kernel looks a handle up at every
//! svc. the kernel makes the handles and the pages are physical, nothing
//! picks keys to collide. the hash is the same every run, and so is the
//! order the maps go in, which saved states keep.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// FxHash's step: the bits so far turned, the next word in, all times an odd
/// constant, whose high bits hashbrown takes.
#[derive(Debug, Default, Clone, Copy)]
pub struct IntHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl IntHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for IntHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
}

pub type BuildIntHasher = BuildHasherDefault<IntHasher>;
pub type IntMap<K, V> = HashMap<K, V, BuildIntHasher>;
pub type IntSet<K> = HashSet<K, BuildIntHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    /// handles in a row, as the kernel makes them, all land and come back.
    #[test]
    fn handles_in_a_row_come_back() {
        let mut map: IntMap<u32, u32> = IntMap::default();
        for handle in 0x1_0000..0x1_4000u32 {
            map.insert(handle, handle ^ 0xFFFF);
        }
        assert_eq!(map.len(), 0x4000);
        assert!((0x1_0000..0x1_4000u32).all(|handle| map.get(&handle) == Some(&(handle ^ 0xFFFF))));
        assert_eq!(map.get(&0x1_4000), None);
    }
}
