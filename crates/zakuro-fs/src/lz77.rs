//! backwards LZSS ("BLZ") decompression, used by the CTR SDK for ExeFS .code
//! when the exheader sets the CompressExefsCode flag.

use crate::FsError;

/// reads the footer to work out how big the decompressed image is.
pub fn decompressed_size(compressed: &[u8]) -> Result<usize, FsError> {
    if compressed.len() < 8 {
        return Err(FsError::BadLz77("buffer shorter than the footer"));
    }
    let add_size = read_u32(compressed, compressed.len() - 4) as usize;
    Ok(compressed.len() + add_size)
}

/// decompresses a BLZ stream.
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>, FsError> {
    let out_size = decompressed_size(compressed)?;
    let compressed_size = compressed.len();

    let footer = read_u32(compressed, compressed_size - 8);
    // high byte, how many bytes at the end are footer + padding.
    let footer_size = ((footer >> 24) & 0xFF) as usize;
    // low 24 bits, distance from the end back to where the payload starts.
    let payload_span = (footer & 0x00FF_FFFF) as usize;

    if footer_size > compressed_size || payload_span > compressed_size {
        return Err(FsError::BadLz77("footer points outside the buffer"));
    }

    let mut out = vec![0u8; out_size];
    out[..compressed_size].copy_from_slice(compressed);

    // index walks backwards through the compressed bytes, out_pos walks
    // backwards through the destination.
    let mut index = compressed_size - footer_size;
    let stop_index = compressed_size - payload_span;
    let mut out_pos = out_size;

    while index > stop_index {
        index -= 1;
        let mut control = compressed[index];

        for _ in 0..8 {
            if index <= stop_index || index == 0 || out_pos == 0 {
                break;
            }

            if control & 0x80 != 0 {
                if index < 2 {
                    return Err(FsError::BadLz77("back-reference truncated"));
                }
                index -= 2;
                let raw = u16::from_le_bytes([compressed[index], compressed[index + 1]]) as usize;
                let run_len = ((raw >> 12) & 0xF) + 3;
                // the stored offset is relative to the current output cursor,
                // biased by 2 because a run is never shorter than that.
                let run_off = (raw & 0x0FFF) + 2;

                if out_pos < run_len {
                    return Err(FsError::BadLz77("run underflows the output"));
                }
                for _ in 0..run_len {
                    let src = out_pos
                        .checked_add(run_off)
                        .filter(|&s| s < out_size)
                        .ok_or(FsError::BadLz77("run reads past the output"))?;
                    let byte = out[src];
                    out_pos -= 1;
                    out[out_pos] = byte;
                }
            } else {
                index -= 1;
                out_pos -= 1;
                out[out_pos] = compressed[index];
            }
            control <<= 1;
        }
    }

    Ok(out)
}

#[inline]
fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}
