//! arrays longer than serde takes, written as sequences

use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// an array of any length, its items one after another
pub fn serialize<S: Serializer, T: Serialize, const N: usize>(array: &[T; N], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(array.iter())
}

pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(deserializer: D) -> Result<[T; N], D::Error> {
    let items = Vec::<T>::deserialize(deserializer)?;
    let length = items.len();
    items.try_into().map_err(|_| D::Error::invalid_length(length, &"as many items as the array has"))
}

/// arrays of arrays of arrays (the DSP's mixes), their innermost items one after another
pub mod nested {
    use super::*;

    pub fn serialize<S: Serializer, T: Serialize, const A: usize, const B: usize, const C: usize>(
        array: &[[[T; C]; B]; A],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(array.iter().flatten().flatten())
    }

    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de> + Copy + Default, const A: usize, const B: usize, const C: usize>(
        deserializer: D,
    ) -> Result<[[[T; C]; B]; A], D::Error> {
        let items = Vec::<T>::deserialize(deserializer)?;
        if items.len() != A * B * C {
            return Err(D::Error::invalid_length(items.len(), &"as many items as the arrays have"));
        }
        let mut array = [[[T::default(); C]; B]; A];
        for (slot, item) in array.iter_mut().flatten().flatten().zip(items) {
            *slot = item;
        }
        Ok(array)
    }
}
