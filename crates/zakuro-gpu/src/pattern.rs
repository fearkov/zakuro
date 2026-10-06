//! bytes that repeat a pattern of a few bytes over and over, as fills leave
//! buffers.

/// the pattern repeated over a block, a whole number of patterns of 2, 3 or 4
/// bytes, and of the vectors the CPU compares and stores at once.
const BLOCK: usize = 48;

/// the blocks looked at together, a few kilobytes, so that bytes differing
/// early stop the comparing early.
const GROUP: usize = 64;

fn block(pattern: &[u8]) -> [u8; BLOCK] {
    debug_assert!(!pattern.is_empty() && BLOCK.is_multiple_of(pattern.len()));
    std::array::from_fn(|i| pattern[i % pattern.len()])
}

/// the bytes compared and written at a time, the pattern over a few
/// kilobytes. the C library compares and copies them in fewer instructions
/// than a loop of 16-byte vectors.
const SPAN: usize = BLOCK * GROUP;

/// the pattern over a span.
fn span(pattern: &[u8]) -> [u8; SPAN] {
    let block = block(pattern);
    let mut span = [0u8; SPAN];
    for chunk in span.as_chunks_mut::<BLOCK>().0 {
        *chunk = block;
    }
    span
}

/// whether bytes hold the pattern over and over, from its first byte.
#[cfg(any(test, feature = "vulkan"))]
pub(crate) fn holds(bytes: &[u8], pattern: &[u8]) -> bool {
    let span = span(pattern);
    bytes.chunks(SPAN).all(|chunk| *chunk == span[..chunk.len()])
}

/// writes the pattern over bytes the way holds reads it, where they do not
/// hold it already. titles clear the same buffers to the same values frame
/// after frame, and reading memory costs less than writing it.
pub(crate) fn fill(bytes: &mut [u8], pattern: &[u8]) {
    let span = span(pattern);
    for chunk in bytes.chunks_mut(SPAN) {
        let span = &span[..chunk.len()];
        if *chunk != *span {
            chunk.copy_from_slice(span);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the pattern over len bytes the plain way.
    fn repeated(pattern: &[u8], len: usize) -> Vec<u8> {
        (0..len).map(|i| pattern[i % pattern.len()]).collect()
    }

    /// a byte changed anywhere, in a group, in the tail or at either end of
    /// either, is seen, and filling leaves the pattern everywhere.
    #[test]
    fn a_byte_off_the_pattern_anywhere_is_seen_and_filled() {
        let lengths = [0, 1, 47, 48, 49, 48 * GROUP - 1, 48 * GROUP, 48 * GROUP + 1, 48 * GROUP * 3 + 29];
        for pattern in [&[0x00, 0x24, 0x24, 0x24][..], &[0x11, 0x22, 0x33], &[0xAB, 0xCD], &[0, 0, 0, 0]] {
            for len in lengths {
                let expected = repeated(pattern, len);
                assert!(holds(&expected, pattern), "{pattern:?} over {len}");
                let mut bytes = expected.clone();
                fill(&mut bytes, pattern);
                assert_eq!(bytes, expected);
                let places = [0, 1, len / 2, len.saturating_sub(48), len.saturating_sub(2), len.saturating_sub(1)];
                for at in places.into_iter().filter(|&at| at < len) {
                    let mut bytes = expected.clone();
                    bytes[at] ^= 0x40;
                    assert!(!holds(&bytes, pattern), "{pattern:?} over {len}, changed at {at}");
                    fill(&mut bytes, pattern);
                    assert_eq!(bytes, expected, "{pattern:?} over {len}, changed at {at}");
                }
                // what something else left there entirely
                let mut bytes: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
                fill(&mut bytes, pattern);
                assert_eq!(bytes, expected);
            }
        }
    }

    /// the pattern starts at the first byte, bytes that start a byte into
    /// it hold it turned around.
    #[test]
    fn the_pattern_starts_at_the_first_byte() {
        let bytes = repeated(&[1, 2, 3, 4], 4096);
        assert!(holds(&bytes[4..], &[1, 2, 3, 4]));
        assert!(!holds(&bytes[1..], &[1, 2, 3, 4]));
        assert!(holds(&bytes[1..], &[2, 3, 4, 1]));
        assert!(!holds(&bytes, &[1, 2]));
    }
}
