use std::fs;
use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CartridgeError {
    #[error("failed to read ROM: {0}")]
    Io(#[from] std::io::Error),
    #[error("ROM is too small to contain an iNES header")]
    TooSmall,
    #[error("missing iNES magic bytes")]
    BadMagic,
    #[error("only mapper 0 / NROM is supported in the first fast path, got mapper {0}")]
    UnsupportedMapper(u8),
}

#[derive(Clone)]
pub struct Cartridge {
    pub prg_rom: Vec<u8>,
    pub chr_rom: Vec<u8>,
    pub vertical_mirroring: bool,
}

impl Cartridge {
    pub fn load_ines(path: impl AsRef<Path>) -> Result<Self, CartridgeError> {
        let data = fs::read(path)?;
        if data.len() < 16 {
            return Err(CartridgeError::TooSmall);
        }
        if &data[0..4] != b"NES\x1a" {
            return Err(CartridgeError::BadMagic);
        }

        let prg_banks = data[4] as usize;
        let chr_banks = data[5] as usize;
        let flags6 = data[6];
        let flags7 = data[7];
        let mapper = (flags6 >> 4) | (flags7 & 0xf0);
        if mapper != 0 {
            return Err(CartridgeError::UnsupportedMapper(mapper));
        }

        let has_trainer = flags6 & 0b0000_0100 != 0;
        let mut offset = 16 + if has_trainer { 512 } else { 0 };
        let prg_len = prg_banks * 16 * 1024;
        let chr_len = chr_banks * 8 * 1024;

        let prg_rom = data[offset..offset + prg_len].to_vec();
        offset += prg_len;
        let chr_rom = if chr_len == 0 {
            vec![0; 8 * 1024]
        } else {
            data[offset..offset + chr_len].to_vec()
        };

        Ok(Self {
            prg_rom,
            chr_rom,
            vertical_mirroring: flags6 & 1 != 0,
        })
    }
}
