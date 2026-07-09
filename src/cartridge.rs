use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

pub const EXPECTED_SMB_ROM_SHA256: &str =
    "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de";

#[derive(Debug, Error)]
pub enum CartridgeError {
    #[error("failed to read ROM: {0}")]
    Io(#[from] std::io::Error),
    #[error("ROM SHA-256 mismatch: got {got}, expected {expected}")]
    DigestMismatch { got: String, expected: &'static str },
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
        assert_expected_rom_digest(&data)?;
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

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn assert_expected_rom_digest(data: &[u8]) -> Result<(), CartridgeError> {
    let digest = sha256_hex(data);
    if digest != EXPECTED_SMB_ROM_SHA256 {
        return Err(CartridgeError::DigestMismatch {
            got: digest,
            expected: EXPECTED_SMB_ROM_SHA256,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_known_digest() {
        assert_eq!(
            sha256_hex(b"SuperMarioBros-Nes-turbo"),
            "f361123acc0af3ad64fae36fb1c7ce16ef0995cccaa5ef2cb7c8b3699518f112"
        );
    }

    #[test]
    fn load_ines_rejects_unexpected_rom_digest() {
        let path = std::env::temp_dir().join(format!(
            "supermariobrosnes-turbo-wrong-rom-{}.nes",
            std::process::id()
        ));
        let mut data = b"NES\x1a".to_vec();
        data.extend([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        data.extend(vec![0; 16 * 1024]);
        data.extend(vec![0; 8 * 1024]);
        fs::write(&path, data).expect("write test ROM");

        let err = match Cartridge::load_ines(&path) {
            Ok(_) => panic!("wrong digest must be rejected"),
            Err(err) => err,
        };
        let _ = fs::remove_file(&path);

        assert!(matches!(err, CartridgeError::DigestMismatch { .. }));
        assert!(err.to_string().contains(EXPECTED_SMB_ROM_SHA256));
    }
}
