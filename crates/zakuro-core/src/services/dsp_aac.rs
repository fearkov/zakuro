//! the audio firmware's AAC decoder, which titles drive through the binary
//! pipe, the third. a request says where ADTS frames are and where each
//! channel's PCM goes, and the answer says how much came out. Pokémon X and
//! Y play their music through it. the layout of the messages is Citra's.

use symphonia_codec_aac::AacDecoder;
use symphonia_core::codecs::audio::well_known::CODEC_ID_AAC;
use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions};
use symphonia_core::packet::PacketRef;
use symphonia_core::units::{Duration, Timestamp};

use crate::System;

/// how long a message is either way.
pub const MESSAGE: usize = 32;

const CODEC_DECODE_AAC: u16 = 1;
const COMMAND_INIT: u16 = 0;
const COMMAND_DECODE: u16 = 1;
const RESULT_ERROR: u32 = 1;

/// where FCRAM starts for the DSP, which addresses it physically.
const FCRAM: u32 = 0x2000_0000;

/// the sample rates ADTS numbers, and how the answer numbers them.
const RATES: [(u32, u32); 9] =
    [(48000, 0), (44100, 1), (32000, 2), (24000, 3), (22050, 4), (16000, 5), (12000, 6), (11025, 7), (8000, 8)];
const ADTS_RATES: [u32; 13] = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

/// the decoder, kept between requests since each frame overlaps the one
/// before, with the stream it was made for.
#[derive(Default)]
pub struct Aac {
    decoder: Option<(AacDecoder, (u32, u32))>,
}

/// a request's answer.
pub fn answer(system: &mut System, request: &[u8]) -> [u8; MESSAGE] {
    let mut message = [0u8; MESSAGE];
    message[..request.len().min(MESSAGE)].copy_from_slice(&request[..request.len().min(MESSAGE)]);
    let half = |at: usize| u16::from_le_bytes([message[at], message[at + 1]]);
    let (codec, command) = (half(0), half(2));
    if codec != CODEC_DECODE_AAC {
        log::warn!("dsp: the binary pipe got codec {codec}, not AAC decoding");
        message[4..8].copy_from_slice(&RESULT_ERROR.to_le_bytes());
        return message;
    }
    // initializing, shutting down and the rest succeed, whatever was left
    // in the result the title sent
    let succeeded = |mut message: [u8; MESSAGE]| {
        message[4..8].copy_from_slice(&0u32.to_le_bytes());
        message
    };
    match command {
        COMMAND_INIT => {
            system.services.dsp.aac.decoder = None;
            succeeded(message)
        }
        COMMAND_DECODE => decode(system, &message),
        _ => succeeded(message),
    }
}

fn decode(system: &mut System, request: &[u8; MESSAGE]) -> [u8; MESSAGE] {
    let word = |at: usize| u32::from_le_bytes(request[at..at + 4].try_into().unwrap());
    let (source, size, left, right) = (word(8), word(12), word(16), word(20));
    let mut answer = [0u8; MESSAGE];
    answer[..4].copy_from_slice(&request[..4]);
    let mut put = |at: usize, value: u32| answer[at..at + 4].copy_from_slice(&value.to_le_bytes());
    // what Citra answers when decoding fails, which keeps titles going
    put(8, 0);
    put(12, 2);
    put(16, size);
    put(28, 1024);

    let mut data = vec![0u8; size.min(0x10_0000) as usize];
    let virtual_source = super::gsp::physical_to_virtual(system, source);
    if source < FCRAM {
        log::warn!("dsp: AAC data at 0x{source:08X}, outside FCRAM");
        return answer;
    }
    system.memory.read_bytes(virtual_source, &mut data);

    let mut channels: Vec<Vec<i16>> = Vec::new();
    let mut stream = (0, 0);
    let mut at = 0;
    while at + 7 <= data.len() {
        let frame = &data[at..];
        // an ADTS header, sync word, then the rate, the channels and the
        // frame's length
        if frame[0] != 0xFF || frame[1] & 0xF0 != 0xF0 {
            log::warn!("dsp: no ADTS frame at {at} of {size} bytes");
            break;
        }
        let protection_absent = frame[1] & 1 != 0;
        let rate_index = ((frame[2] >> 2) & 0xF) as usize;
        let layout = (((frame[2] & 1) << 2) | (frame[3] >> 6)) as u32;
        let length = ((((frame[3] & 3) as usize) << 11) | ((frame[4] as usize) << 3) | ((frame[5] as usize) >> 5)).min(frame.len());
        let header = if protection_absent { 7 } else { 9 };
        let Some(&rate) = ADTS_RATES.get(rate_index) else { break };
        if length <= header {
            break;
        }
        stream = (rate, layout);
        let decoder = decoder_for(&mut system.services.dsp.aac, rate_index as u32, layout);
        let Some(decoder) = decoder else { break };
        let packet = PacketRef::new(0, Timestamp::ZERO, Duration::new(1024), &frame[header..length]);
        match decoder.decode_ref(&packet) {
            Ok(decoded) => {
                let mut planes: Vec<Vec<i16>> = Vec::new();
                decoded.copy_to_vecs_planar::<i16>(&mut planes);
                channels.resize(planes.len().max(channels.len()), Vec::new());
                for (channel, plane) in channels.iter_mut().zip(planes) {
                    channel.extend(plane);
                }
            }
            Err(error) => log::warn!("dsp: an AAC frame did not decode, {error}"),
        }
        at += length;
    }
    if channels.is_empty() {
        return answer;
    }

    for (channel, target) in channels.iter().zip([left, right]) {
        if target < FCRAM {
            continue;
        }
        let bytes: Vec<u8> = channel.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        let address = super::gsp::physical_to_virtual(system, target);
        system.write_from_service(address, &bytes);
    }
    let rate = RATES.iter().find(|(hz, _)| *hz == stream.0).map_or(0, |&(_, index)| index);
    put(8, rate);
    put(12, channels.len() as u32);
    put(28, channels[0].len() as u32);
    answer
}

/// the decoder for a stream, made again when the stream changes.
fn decoder_for(aac: &mut Aac, rate_index: u32, layout: u32) -> Option<&mut AacDecoder> {
    let key = (rate_index, layout);
    if aac.decoder.as_ref().is_none_or(|(_, made)| *made != key) {
        // the audio specific config, low complexity at this rate and layout
        let config = (2u16 << 11) | ((rate_index as u16) << 7) | ((layout as u16) << 3);
        let mut params = AudioCodecParameters::new();
        params.for_codec(CODEC_ID_AAC).with_extra_data(config.to_be_bytes().to_vec().into_boxed_slice());
        match AacDecoder::try_new(&params, &AudioDecoderOptions::default()) {
            Ok(decoder) => aac.decoder = Some((decoder, key)),
            Err(error) => {
                log::warn!("dsp: no AAC decoder for rate {rate_index} and layout {layout}, {error}");
                aac.decoder = None;
            }
        }
    }
    aac.decoder.as_mut().map(|(decoder, _)| decoder)
}
