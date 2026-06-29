use crate::cartridge::Cartridge;
use std::sync::OnceLock;
use thiserror::Error;

pub const NES_WIDTH: usize = 256;
pub const NES_HEIGHT: usize = 240;
pub const RGB_CHANNELS: usize = 3;
pub const FRAME_PIXELS_RGB: usize = NES_WIDTH * NES_HEIGHT * RGB_CHANNELS;
pub const FRAME_PIXELS_GRAY: usize = NES_WIDTH * NES_HEIGHT;

const CPU_CYCLES_PER_FRAME_GUARD: usize = 40_000;
const PPU_DOTS_PER_SCANLINE: usize = 341;
const PPU_SCANLINES_PER_FRAME: usize = 262;
const PPU_DOTS_PER_FRAME: usize = PPU_DOTS_PER_SCANLINE * PPU_SCANLINES_PER_FRAME;
const PPU_SPRITE0_DOT: usize = 30 * PPU_DOTS_PER_SCANLINE + 1;
const PPU_VBLANK_DOT: usize = 241 * PPU_DOTS_PER_SCANLINE + 1;
const PPU_PRERENDER_DOT: usize = 261 * PPU_DOTS_PER_SCANLINE + 1;
const DEFAULT_GRAY_CROP_TOP: usize = 32;
const DEFAULT_GRAY_CROP_HEIGHT: usize = NES_HEIGHT - DEFAULT_GRAY_CROP_TOP;
const DEFAULT_GRAY_RESIZE_WIDTH: usize = 84;
const DEFAULT_GRAY_RESIZE_HEIGHT: usize = 84;
const DEFAULT_GRAY_RESIZE_PIXELS: usize = DEFAULT_GRAY_RESIZE_WIDTH * DEFAULT_GRAY_RESIZE_HEIGHT;
const DEFAULT_GRAY_AREA_DENOM: u32 = (NES_WIDTH * DEFAULT_GRAY_CROP_HEIGHT) as u32;
const SPRITE_SHADOW_EMPTY: u8 = 255;

const FLAG_C: u8 = 0x01;
const FLAG_Z: u8 = 0x02;
const FLAG_I: u8 = 0x04;
const FLAG_D: u8 = 0x08;
const FLAG_B: u8 = 0x10;
const FLAG_U: u8 = 0x20;
const FLAG_V: u8 = 0x40;
const FLAG_N: u8 = 0x80;

const BUTTON_A: u8 = 1 << 0;
const BUTTON_B: u8 = 1 << 1;
const BUTTON_START: u8 = 1 << 3;
const BUTTON_LEFT: u8 = 1 << 6;
const BUTTON_RIGHT: u8 = 1 << 7;

#[derive(Debug, Error)]
pub enum StateLoadError {
    #[error("state field {name} with size {size} was not found")]
    MissingField { name: &'static str, size: usize },
}

#[derive(Clone, Copy, Debug)]
pub enum MarioAction {
    Noop = 0,
    Right = 1,
    RightB = 2,
    RightA = 3,
    RightAB = 4,
    A = 5,
    Left = 6,
    Start = 7,
}

impl MarioAction {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Right,
            2 => Self::RightB,
            3 => Self::RightA,
            4 => Self::RightAB,
            5 => Self::A,
            6 => Self::Left,
            7 => Self::Start,
            _ => Self::Noop,
        }
    }

    #[inline]
    fn buttons(self) -> u8 {
        match self {
            Self::Noop => 0,
            Self::Right => BUTTON_RIGHT,
            Self::RightB => BUTTON_RIGHT | BUTTON_B,
            Self::RightA => BUTTON_RIGHT | BUTTON_A,
            Self::RightAB => BUTTON_RIGHT | BUTTON_A | BUTTON_B,
            Self::A => BUTTON_A,
            Self::Left => BUTTON_LEFT,
            Self::Start => BUTTON_START,
        }
    }
}

#[derive(Clone, Copy)]
struct Cpu {
    a: u8,
    x: u8,
    y: u8,
    sp: u8,
    pc: u16,
    p: u8,
}

impl Cpu {
    fn new() -> Self {
        Self {
            a: 0,
            x: 0,
            y: 0,
            sp: 0xfd,
            pc: 0,
            p: FLAG_U | FLAG_I,
        }
    }
}

#[derive(Clone)]
struct Ppu {
    chr_rom: Vec<u8>,
    chr_addr_mask: usize,
    vertical_mirroring: bool,
    ctrl: u8,
    mask: u8,
    status: u8,
    oam_addr: u8,
    oam: [u8; 256],
    vram: [u8; 2048],
    palette: [u8; 32],
    data_buffer: u8,
    addr: u16,
    temp_addr: u16,
    scroll_addr: u16,
    render_addr: u16,
    first_write: bool,
    fine_x: u8,
    scroll_x_px: u16,
    scroll_y_px: u16,
    scroll_x_low: u8,
    scroll_override_x_px: Option<u16>,
    frame_dot: usize,
    frame: u64,
    nmi_pending: bool,
}

#[derive(Clone, Copy)]
struct Area84AxisContribution {
    dst: u16,
    weight: u16,
}

struct DefaultGrayArea84Plan {
    x: [[Area84AxisContribution; 2]; NES_WIDTH],
    y: [[Area84AxisContribution; 2]; DEFAULT_GRAY_CROP_HEIGHT],
}

fn default_gray_area84_plan() -> &'static DefaultGrayArea84Plan {
    static PLAN: OnceLock<DefaultGrayArea84Plan> = OnceLock::new();
    PLAN.get_or_init(|| DefaultGrayArea84Plan {
        x: build_area84_inverse_axis::<NES_WIDTH, DEFAULT_GRAY_RESIZE_WIDTH>(),
        y: build_area84_inverse_axis::<DEFAULT_GRAY_CROP_HEIGHT, DEFAULT_GRAY_RESIZE_HEIGHT>(),
    })
}

fn build_area84_inverse_axis<const SRC: usize, const DST: usize>(
) -> [[Area84AxisContribution; 2]; SRC] {
    let empty = Area84AxisContribution { dst: 0, weight: 0 };
    let mut out = [[empty; 2]; SRC];
    for src_i in 0..SRC {
        let src_start = src_i * DST;
        let src_end = (src_i + 1) * DST;
        let first_dst = src_start / SRC;
        let last_dst_exclusive = (src_end + SRC - 1) / SRC;
        for (slot, dst_i) in (first_dst..last_dst_exclusive).enumerate() {
            let dst_start = dst_i * SRC;
            let dst_end = (dst_i + 1) * SRC;
            let weight = src_end
                .min(dst_end)
                .saturating_sub(src_start.max(dst_start));
            out[src_i][slot] = Area84AxisContribution {
                dst: dst_i as u16,
                weight: weight as u16,
            };
        }
    }
    out
}

#[inline]
fn accumulate_area_84x84_pixel(
    sums: &mut [u32; DEFAULT_GRAY_RESIZE_PIXELS],
    src_x: usize,
    src_y: usize,
    gray: u8,
) {
    let plan = default_gray_area84_plan();
    let value = gray as u32;
    for y_contribution in plan.y[src_y] {
        if y_contribution.weight == 0 {
            continue;
        }
        let row = y_contribution.dst as usize * DEFAULT_GRAY_RESIZE_WIDTH;
        let y_weight = y_contribution.weight as u32;
        for x_contribution in plan.x[src_x] {
            if x_contribution.weight == 0 {
                continue;
            }
            sums[row + x_contribution.dst as usize] +=
                value * y_weight * x_contribution.weight as u32;
        }
    }
}

#[inline]
fn add_area_84x84_delta(
    sums: &mut [u32; DEFAULT_GRAY_RESIZE_PIXELS],
    src_x: usize,
    src_y: usize,
    delta: i16,
) {
    let plan = default_gray_area84_plan();
    let magnitude = delta.unsigned_abs() as u32;
    for y_contribution in plan.y[src_y] {
        if y_contribution.weight == 0 {
            continue;
        }
        let row = y_contribution.dst as usize * DEFAULT_GRAY_RESIZE_WIDTH;
        let y_weight = y_contribution.weight as u32;
        for x_contribution in plan.x[src_x] {
            if x_contribution.weight == 0 {
                continue;
            }
            let weighted = magnitude * y_weight * x_contribution.weight as u32;
            let dst = &mut sums[row + x_contribution.dst as usize];
            if delta >= 0 {
                *dst += weighted;
            } else {
                *dst -= weighted;
            }
        }
    }
}

impl Ppu {
    fn new(chr_rom: Vec<u8>, vertical_mirroring: bool) -> Self {
        let chr_addr_mask = chr_rom.len() - 1;
        Self {
            chr_rom,
            chr_addr_mask,
            vertical_mirroring,
            ctrl: 0,
            mask: 0,
            status: 0,
            oam_addr: 0,
            oam: [0; 256],
            vram: [0; 2048],
            palette: [0; 32],
            data_buffer: 0,
            addr: 0,
            temp_addr: 0,
            scroll_addr: 0,
            render_addr: 0,
            first_write: true,
            fine_x: 0,
            scroll_x_px: 0,
            scroll_y_px: 0,
            scroll_x_low: 0,
            scroll_override_x_px: None,
            frame_dot: 0,
            frame: 0,
            nmi_pending: false,
        }
    }

    fn reset(&mut self) {
        self.ctrl = 0;
        self.mask = 0;
        self.status = 0;
        self.oam_addr = 0;
        self.oam = [0; 256];
        self.vram = [0; 2048];
        self.palette = [0; 32];
        self.data_buffer = 0;
        self.addr = 0;
        self.temp_addr = 0;
        self.scroll_addr = 0;
        self.render_addr = 0;
        self.first_write = true;
        self.fine_x = 0;
        self.scroll_x_px = 0;
        self.scroll_y_px = 0;
        self.scroll_x_low = 0;
        self.scroll_override_x_px = None;
        self.frame_dot = 0;
        self.frame = 0;
        self.nmi_pending = false;
    }

    fn load_fceu_state(
        &mut self,
        ntar: &[u8],
        pram: &[u8],
        spra: &[u8],
        ppur: &[u8],
        radd: Option<&[u8]>,
        tadd: Option<&[u8]>,
        xoff: Option<&[u8]>,
    ) {
        self.vram.copy_from_slice(ntar);
        self.palette.copy_from_slice(pram);
        self.oam.copy_from_slice(spra);
        self.ctrl = ppur[0];
        self.mask = ppur[1];
        self.status = ppur[2];
        self.oam_addr = ppur[3];
        self.addr = radd.and_then(read_u16_le).unwrap_or(0);
        self.temp_addr = tadd.and_then(read_u16_le).unwrap_or(0);
        self.scroll_addr = self.temp_addr;
        self.render_addr = self.addr;
        self.first_write = true;
        self.fine_x = xoff.and_then(|value| value.first().copied()).unwrap_or(0);
        self.frame_dot = 0;
        self.nmi_pending = false;
        self.update_scroll_x_px();
    }

    #[inline]
    fn tick(&mut self, ppu_cycles: usize) -> bool {
        let mut completed_frame = false;
        let mut remaining = ppu_cycles;
        while remaining > 0 {
            let current = self.dot();
            let next = next_ppu_event_dot(current);
            let advance = remaining.min(next - current);
            self.set_dot(current + advance);
            remaining -= advance;

            match self.dot() {
                PPU_SPRITE0_DOT => self.status |= 0x40,
                PPU_VBLANK_DOT => {
                    self.status |= 0x80;
                    if self.ctrl & 0x80 != 0 {
                        self.nmi_pending = true;
                    }
                }
                PPU_PRERENDER_DOT => {
                    self.status &= !0xc0;
                }
                PPU_DOTS_PER_FRAME => {
                    self.frame_dot = 0;
                    self.frame = self.frame.wrapping_add(1);
                    completed_frame = true;
                }
                _ => {}
            }
        }
        completed_frame
    }

    #[inline]
    fn dot(&self) -> usize {
        self.frame_dot
    }

    #[inline]
    fn set_dot(&mut self, dot: usize) {
        self.frame_dot = dot;
    }

    #[inline]
    fn take_nmi(&mut self) -> bool {
        let pending = self.nmi_pending;
        self.nmi_pending = false;
        pending
    }

    #[inline]
    fn cpu_read_register(&mut self, reg: u16) -> u8 {
        match reg & 7 {
            2 => {
                let value = self.status;
                self.status &= !0x80;
                self.first_write = true;
                value
            }
            4 => self.oam[self.oam_addr as usize],
            7 => self.read_data(),
            _ => 0,
        }
    }

    #[inline]
    fn cpu_write_register(&mut self, reg: u16, value: u8) {
        match reg & 7 {
            0 => {
                let old = self.ctrl;
                self.ctrl = value;
                self.temp_addr = (self.temp_addr & 0xf3ff) | (((value as u16) & 0x03) << 10);
                self.scroll_addr = (self.scroll_addr & 0xf3ff) | (((value as u16) & 0x03) << 10);
                self.update_scroll_x_px();
                if old & 0x80 == 0 && value & 0x80 != 0 && self.status & 0x80 != 0 {
                    self.nmi_pending = true;
                }
            }
            1 => self.mask = value,
            3 => self.oam_addr = value,
            4 => {
                self.oam[self.oam_addr as usize] = value;
                self.oam_addr = self.oam_addr.wrapping_add(1);
            }
            5 => {
                if self.first_write {
                    self.fine_x = value & 0x07;
                    self.scroll_x_low = value;
                    self.update_scroll_x_px();
                    self.temp_addr = (self.temp_addr & 0xffe0) | ((value as u16) >> 3);
                } else {
                    self.scroll_y_px = value as u16;
                    self.temp_addr = (self.temp_addr & 0x8fff) | (((value as u16) & 0x07) << 12);
                    self.temp_addr = (self.temp_addr & 0xfc1f) | (((value as u16) & 0xf8) << 2);
                }
                self.scroll_addr = self.temp_addr;
                self.first_write = !self.first_write;
            }
            6 => {
                if self.first_write {
                    self.temp_addr = (self.temp_addr & 0x00ff) | (((value as u16) & 0x3f) << 8);
                } else {
                    self.temp_addr = (self.temp_addr & 0xff00) | value as u16;
                    self.addr = self.temp_addr;
                }
                self.first_write = !self.first_write;
            }
            7 => self.write_data(value),
            _ => {}
        }
    }

    #[inline]
    fn read_data(&mut self) -> u8 {
        let addr = self.addr & 0x3fff;
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;

        if addr >= 0x3f00 {
            self.ppu_read(addr)
        } else {
            let buffered = self.data_buffer;
            self.data_buffer = self.ppu_read(addr);
            buffered
        }
    }

    #[inline]
    fn write_data(&mut self, value: u8) {
        let addr = self.addr & 0x3fff;
        self.ppu_write(addr, value);
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;
    }

    #[inline]
    fn chr_read(&self, addr: usize) -> u8 {
        let idx = addr & self.chr_addr_mask;
        // SAFETY: SMB/NROM CHR ROM sizes are power-of-two and chr_addr_mask is len - 1.
        unsafe { *self.chr_rom.get_unchecked(idx) }
    }

    #[inline]
    fn ppu_read(&self, addr: u16) -> u8 {
        let addr = addr & 0x3fff;
        match addr {
            0x0000..=0x1fff => self.chr_read(addr as usize),
            0x2000..=0x3eff => {
                let idx = self.mirror_nametable_addr(addr);
                self.vram[idx]
            }
            0x3f00..=0x3fff => self.palette[self.mirror_palette_addr(addr)],
            _ => 0,
        }
    }

    #[inline]
    fn ppu_write(&mut self, addr: u16, value: u8) {
        let addr = addr & 0x3fff;
        match addr {
            0x0000..=0x1fff => {}
            0x2000..=0x3eff => {
                let idx = self.mirror_nametable_addr(addr);
                self.vram[idx] = value;
            }
            0x3f00..=0x3fff => {
                let idx = self.mirror_palette_addr(addr);
                self.palette[idx] = value;
            }
            _ => {}
        }
    }

    #[inline]
    fn mirror_nametable_addr(&self, addr: u16) -> usize {
        let v = (addr - 0x2000) as usize % 0x1000;
        let table = v / 0x400;
        let offset = v & 0x3ff;
        let physical_table = if self.vertical_mirroring {
            table & 1
        } else {
            (table >> 1) & 1
        };
        physical_table * 0x400 + offset
    }

    #[inline]
    fn mirror_palette_addr(&self, addr: u16) -> usize {
        let mut idx = (addr as usize - 0x3f00) & 0x1f;
        if idx == 0x10 {
            idx = 0x00;
        } else if idx == 0x14 {
            idx = 0x04;
        } else if idx == 0x18 {
            idx = 0x08;
        } else if idx == 0x1c {
            idx = 0x0c;
        }
        idx
    }

    fn write_gray_frame(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), FRAME_PIXELS_GRAY);
        for y in 0..NES_HEIGHT {
            for x in 0..NES_WIDTH {
                let color = self.bg_color_index(x, y);
                dst[y * NES_WIDTH + x] = NES_GRAY_PALETTE[color as usize];
            }
        }
        self.draw_sprites_gray(dst);
    }

    fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height);
        self.write_bg_gray_cropped_tiled(dst, crop_top, height);
        self.draw_sprites_gray_cropped(dst, crop_top, height);
    }

    fn write_gray_frame_cropped_area_84x84(&self, dst: &mut [u8], sprite_shadow: &mut [u8]) {
        debug_assert_eq!(dst.len(), DEFAULT_GRAY_RESIZE_PIXELS);
        debug_assert!(sprite_shadow.len() >= NES_WIDTH * DEFAULT_GRAY_CROP_HEIGHT);

        let mut sums = [0u32; DEFAULT_GRAY_RESIZE_PIXELS];
        let sprite_shadow = &mut sprite_shadow[..NES_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];
        sprite_shadow.fill(SPRITE_SHADOW_EMPTY);

        self.accumulate_bg_gray_area_84x84(&mut sums);
        self.accumulate_sprite_gray_area_84x84_deltas(&mut sums, sprite_shadow);

        let rounding = DEFAULT_GRAY_AREA_DENOM / 2;
        for (dst, &sum) in dst.iter_mut().zip(sums.iter()) {
            *dst = ((sum + rounding) / DEFAULT_GRAY_AREA_DENOM) as u8;
        }
    }

    fn write_bg_gray_cropped_tiled(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            dst.fill(palette_gray[0]);
            return;
        }

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..height {
            let y = crop_top + out_y;
            let world_y = if y < 32 { y } else { y + scroll_y };
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let row_start = out_y * NES_WIDTH;
            let mut x = 0usize;

            while x < NES_WIDTH {
                let world_x = if y < 32 { x } else { x + scroll_x };
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let nt_base = 0x2000 + (table as u16) * 0x400;
                let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
                let attr =
                    self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                let run = (8 - fine_x).min(NES_WIDTH - x);

                for col in 0..run {
                    let bit = 7 - (fine_x + col);
                    let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                    let gray = if pixel == 0 {
                        palette_gray[0]
                    } else {
                        palette_gray[(palette_id as usize) * 4 + pixel as usize]
                    };
                    dst[row_start + x + col] = gray;
                }

                x += run;
            }
        }
    }

    fn accumulate_bg_gray_area_84x84(&self, sums: &mut [u32; DEFAULT_GRAY_RESIZE_PIXELS]) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            for out_y in 0..DEFAULT_GRAY_CROP_HEIGHT {
                for x in 0..NES_WIDTH {
                    accumulate_area_84x84_pixel(sums, x, out_y, palette_gray[0]);
                }
            }
            return;
        }

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..DEFAULT_GRAY_CROP_HEIGHT {
            let y = DEFAULT_GRAY_CROP_TOP + out_y;
            let world_y = if y < 32 { y } else { y + scroll_y };
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let mut x = 0usize;

            while x < NES_WIDTH {
                let world_x = if y < 32 { x } else { x + scroll_x };
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let nt_base = 0x2000 + (table as u16) * 0x400;
                let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
                let attr =
                    self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                let run = (8 - fine_x).min(NES_WIDTH - x);

                for col in 0..run {
                    let bit = 7 - (fine_x + col);
                    let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                    let gray = if pixel == 0 {
                        palette_gray[0]
                    } else {
                        palette_gray[(palette_id as usize) * 4 + pixel as usize]
                    };
                    accumulate_area_84x84_pixel(sums, x + col, out_y, gray);
                }

                x += run;
            }
        }
    }

    fn write_rgb_frame(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), FRAME_PIXELS_RGB);
        let plane = NES_WIDTH * NES_HEIGHT;
        for y in 0..NES_HEIGHT {
            for x in 0..NES_WIDTH {
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = y * NES_WIDTH + x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb(dst);
    }

    fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height * RGB_CHANNELS);
        let plane = NES_WIDTH * height;
        for out_y in 0..height {
            let y = crop_top + out_y;
            for x in 0..NES_WIDTH {
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = out_y * NES_WIDTH + x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb_cropped(dst, crop_top, height);
    }

    #[inline]
    fn bg_color_index(&self, x: usize, y: usize) -> u8 {
        if self.mask & 0x08 == 0 {
            return self.palette[0];
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let attr = self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
        let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
        let palette_id = (attr >> shift) & 0x03;

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
        if pixel == 0 {
            self.palette[0]
        } else {
            self.palette[(palette_id as usize) * 4 + pixel as usize]
        }
    }

    #[inline]
    fn bg_pixel_opaque(&self, x: usize, y: usize) -> bool {
        if self.mask & 0x08 == 0 {
            return false;
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        (((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)) != 0
    }

    #[inline]
    fn bg_gray_pixel(&self, x: usize, y: usize, palette_gray: &[u8; 32]) -> u8 {
        if self.mask & 0x08 == 0 {
            return palette_gray[0];
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let attr = self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
        let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
        let palette_id = (attr >> shift) & 0x03;

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
        if pixel == 0 {
            palette_gray[0]
        } else {
            palette_gray[(palette_id as usize) * 4 + pixel as usize]
        }
    }

    fn draw_sprites_gray(&self, dst: &mut [u8]) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        continue;
                    }
                    dst[screen_y as usize * NES_WIDTH + screen_x as usize] =
                        palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    fn draw_sprites_gray_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let crop_top = crop_top as i16;
        let crop_bottom = crop_top + height as i16;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        continue;
                    }
                    let out_y = (screen_y - crop_top) as usize;
                    dst[out_y * NES_WIDTH + screen_x as usize] =
                        palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    fn accumulate_sprite_gray_area_84x84_deltas(
        &self,
        sums: &mut [u32; DEFAULT_GRAY_RESIZE_PIXELS],
        sprite_shadow: &mut [u8],
    ) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let crop_top = DEFAULT_GRAY_CROP_TOP as i16;
        let crop_bottom = crop_top + DEFAULT_GRAY_CROP_HEIGHT as i16;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let x = screen_x as usize;
                    let y = screen_y as usize;
                    if behind_background && self.bg_pixel_opaque(x, y) {
                        continue;
                    }

                    let out_y = y - DEFAULT_GRAY_CROP_TOP;
                    let shadow_idx = out_y * NES_WIDTH + x;
                    let old_gray = if sprite_shadow[shadow_idx] == SPRITE_SHADOW_EMPTY {
                        self.bg_gray_pixel(x, y, &palette_gray)
                    } else {
                        sprite_shadow[shadow_idx]
                    };
                    let new_gray = palette_gray[palette_base + pixel as usize];
                    if new_gray != old_gray {
                        add_area_84x84_delta(sums, x, out_y, new_gray as i16 - old_gray as i16);
                        sprite_shadow[shadow_idx] = new_gray;
                    }
                }
            }
        }
    }

    fn draw_sprites_rgb(&self, dst: &mut [u8]) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let plane = NES_WIDTH * NES_HEIGHT;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    let idx = screen_y as usize * NES_WIDTH + screen_x as usize;
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    fn draw_sprites_rgb_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let crop_top = crop_top as i16;
        let crop_bottom = crop_top + height as i16;
        let plane = NES_WIDTH * height;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    let out_y = (screen_y - crop_top) as usize;
                    let idx = out_y * NES_WIDTH + screen_x as usize;
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    #[inline]
    fn update_scroll_x_px(&mut self) {
        self.scroll_x_px = (((self.ctrl & 0x01) as u16) << 8) | self.scroll_x_low as u16;
    }

    #[inline]
    fn set_scroll_override_x(&mut self, scroll_x_px: Option<u16>) {
        self.scroll_override_x_px = scroll_x_px;
    }

    #[inline]
    fn render_scroll_x_px(&self) -> u16 {
        self.scroll_override_x_px.unwrap_or(self.scroll_x_px)
    }

    #[inline]
    fn palette_gray(&self) -> [u8; 32] {
        let mut out = [0; 32];
        for (dst, &color) in out.iter_mut().zip(self.palette.iter()) {
            *dst = NES_GRAY_PALETTE[color as usize];
        }
        out
    }

    #[inline]
    fn bg_world_pos(&self, x: usize, y: usize) -> (usize, usize) {
        if y < 32 {
            (x, y)
        } else {
            (
                x + self.render_scroll_x_px() as usize,
                y + self.scroll_y_px as usize,
            )
        }
    }
}

const NES_GRAY_PALETTE: [u8; 64] = build_nes_gray_palette();

const fn build_nes_gray_palette() -> [u8; 64] {
    let mut table = [0; 64];
    let mut color = 0;
    while color < 64 {
        let hue = (color & 0x0f) as u8;
        let level = ((color >> 4) & 0x03) as u8;
        table[color] = level * 56 + hue * 5;
        color += 1;
    }
    table
}

#[inline]
fn nes_rgb(color: u8) -> [u8; 3] {
    NES_RGB_PALETTE[(color as usize) & 0x3f]
}

const NES_RGB_PALETTE: [[u8; 3]; 64] = [
    [84, 84, 84],
    [0, 30, 116],
    [8, 16, 144],
    [48, 0, 136],
    [68, 0, 100],
    [92, 0, 48],
    [84, 4, 0],
    [60, 24, 0],
    [32, 42, 0],
    [8, 58, 0],
    [0, 64, 0],
    [0, 60, 0],
    [0, 50, 60],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [152, 150, 152],
    [8, 76, 196],
    [48, 50, 236],
    [92, 30, 228],
    [136, 20, 176],
    [160, 20, 100],
    [152, 34, 32],
    [120, 60, 0],
    [84, 90, 0],
    [40, 114, 0],
    [8, 124, 0],
    [0, 118, 40],
    [0, 102, 120],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [236, 238, 236],
    [76, 154, 236],
    [120, 124, 236],
    [176, 98, 236],
    [228, 84, 236],
    [236, 88, 180],
    [236, 106, 100],
    [212, 136, 32],
    [160, 170, 0],
    [116, 196, 0],
    [76, 208, 32],
    [56, 204, 108],
    [56, 180, 204],
    [60, 60, 60],
    [0, 0, 0],
    [0, 0, 0],
    [236, 238, 236],
    [168, 204, 236],
    [188, 188, 236],
    [212, 178, 236],
    [236, 174, 236],
    [236, 174, 212],
    [236, 180, 176],
    [228, 196, 144],
    [204, 210, 120],
    [180, 222, 120],
    [168, 226, 144],
    [152, 226, 180],
    [160, 214, 228],
    [160, 162, 160],
    [0, 0, 0],
    [0, 0, 0],
];

#[inline]
fn next_ppu_event_dot(current: usize) -> usize {
    if current < PPU_SPRITE0_DOT {
        PPU_SPRITE0_DOT
    } else if current < PPU_VBLANK_DOT {
        PPU_VBLANK_DOT
    } else if current < PPU_PRERENDER_DOT {
        PPU_PRERENDER_DOT
    } else {
        PPU_DOTS_PER_FRAME
    }
}

fn required_field<'a>(
    state: &'a [u8],
    name: &'static [u8; 4],
    display_name: &'static str,
    size: usize,
) -> Result<&'a [u8], StateLoadError> {
    optional_field(state, name, size).ok_or(StateLoadError::MissingField {
        name: display_name,
        size,
    })
}

fn optional_field<'a>(state: &'a [u8], name: &[u8; 4], size: usize) -> Option<&'a [u8]> {
    let header_len = name.len() + 4;
    state
        .windows(header_len)
        .enumerate()
        .find_map(|(offset, window)| {
            if &window[..4] != name {
                return None;
            }
            let field_size = u32::from_le_bytes(window[4..8].try_into().ok()?) as usize;
            if field_size != size {
                return None;
            }
            let start = offset + header_len;
            let end = start.checked_add(field_size)?;
            state.get(start..end)
        })
}

fn read_u16_le(value: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(value.get(..2)?.try_into().ok()?))
}

#[derive(Clone)]
pub struct NesEmulator {
    cpu: Cpu,
    ppu: Ppu,
    ram: [u8; 2048],
    prg_rom: Vec<u8>,
    prg_addr_mask: usize,
    controller_state: u8,
    controller_shift: u8,
    controller_strobe: bool,
    extra_cycles: u16,
    x_pos: u16,
    lives: u8,
    terminate_on_flag: bool,
    done: bool,
}

impl NesEmulator {
    pub fn new_with_options(cart: Cartridge, terminate_on_flag: bool) -> Self {
        let prg_addr_mask = cart.prg_rom.len() - 1;
        let ppu = Ppu::new(cart.chr_rom, cart.vertical_mirroring);
        let mut emu = Self {
            cpu: Cpu::new(),
            ppu,
            ram: [0; 2048],
            prg_rom: cart.prg_rom,
            prg_addr_mask,
            controller_state: 0,
            controller_shift: 0,
            controller_strobe: false,
            extra_cycles: 0,
            x_pos: 0,
            lives: 0,
            terminate_on_flag,
            done: false,
        };
        emu.reset();
        emu
    }

    pub fn reset(&mut self) {
        self.cpu = Cpu::new();
        self.ppu.reset();
        self.ram = [0; 2048];
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        self.done = false;
        self.cpu.pc = self.cpu_read_u16(0xfffc);
        self.refresh_smb_state();
    }

    pub fn load_fceu_state(&mut self, state: &[u8]) -> Result<(), StateLoadError> {
        let pc = required_field(state, b"PC\0\0", "PC", 2)?;
        let a = required_field(state, b"A\0\0\0", "A", 1)?[0];
        let x = required_field(state, b"X\0\0\0", "X", 1)?[0];
        let y = required_field(state, b"Y\0\0\0", "Y", 1)?[0];
        let sp = required_field(state, b"S\0\0\0", "S", 1)?[0];
        let p = required_field(state, b"P\0\0\0", "P", 1)?[0];
        let ram = required_field(state, b"RAM\0", "RAM", 2048)?;
        let ntar = required_field(state, b"NTAR", "NTAR", 2048)?;
        let pram = required_field(state, b"PRAM", "PRAM", 32)?;
        let spra = required_field(state, b"SPRA", "SPRA", 256)?;
        let ppur = required_field(state, b"PPUR", "PPUR", 4)?;

        self.cpu = Cpu {
            a,
            x,
            y,
            sp,
            pc: read_u16_le(pc).unwrap_or(0),
            p,
        };
        self.ppu.load_fceu_state(
            ntar,
            pram,
            spra,
            ppur,
            optional_field(state, b"RADD", 2),
            optional_field(state, b"TADD", 2),
            optional_field(state, b"XOFF", 1),
        );
        self.ram.copy_from_slice(ram);
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        self.done = false;
        self.refresh_smb_state();
        Ok(())
    }

    #[inline]
    pub fn step_frame(&mut self, action: MarioAction) -> f32 {
        if self.done {
            return 0.0;
        }

        let before = self.x_pos;
        self.run_frame(action.buttons());
        self.refresh_smb_state();
        if self.terminate_on_flag && self.x_pos >= 3160 {
            self.done = true;
        }
        self.x_pos.saturating_sub(before) as f32
    }

    #[inline]
    pub fn write_rgb_frame(&self, dst: &mut [u8]) {
        self.ppu.write_rgb_frame(dst);
    }

    #[inline]
    pub fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_rgb_frame_cropped(dst, crop_top, height);
    }

    #[inline]
    pub fn write_gray_frame(&self, dst: &mut [u8]) {
        self.ppu.write_gray_frame(dst);
    }

    #[inline]
    pub fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_gray_frame_cropped(dst, crop_top, height);
    }

    #[inline]
    pub fn write_gray_frame_cropped_area_84x84(&self, dst: &mut [u8], sprite_shadow: &mut [u8]) {
        self.ppu
            .write_gray_frame_cropped_area_84x84(dst, sprite_shadow);
    }

    #[inline]
    pub fn x_pos(&self) -> u16 {
        self.x_pos
    }

    #[inline]
    pub fn lives(&self) -> u8 {
        self.lives
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.done
    }

    fn run_frame(&mut self, buttons: u8) {
        self.controller_state = buttons;
        let mut cpu_cycle_guard = 0usize;
        loop {
            if self.ppu.take_nmi() {
                self.interrupt(0xfffa, false);
            }
            let cycles = self.cpu_step() as usize;
            cpu_cycle_guard += cycles;
            if self.ppu.tick(cycles * 3) || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD {
                break;
            }
        }
    }

    #[inline]
    fn refresh_smb_state(&mut self) {
        self.x_pos = ((self.ram[0x006d] as u16) << 8) | self.ram[0x0086] as u16;
        self.lives = self.ram[0x075a];
        let active_gameplay = self.ram[0x0770] == 1 && self.ram[0x0772] == 3;
        let scroll_x = ((self.ram[0x071a] as u16) << 8) | self.ram[0x071c] as u16;
        self.ppu
            .set_scroll_override_x(active_gameplay.then_some(scroll_x));
    }

    #[inline]
    fn cpu_read(&mut self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff],
            0x2000..=0x3fff => self.ppu.cpu_read_register(addr),
            0x4016 => self.controller_read(),
            0x8000..=0xffff => self.prg_read(addr),
            _ => 0,
        }
    }

    #[inline]
    fn cpu_write(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff] = value,
            0x2000..=0x3fff => self.ppu.cpu_write_register(addr, value),
            0x4014 => self.oam_dma(value),
            0x4016 => self.controller_write(value),
            _ => {}
        }
    }

    #[inline]
    fn prg_read(&self, addr: u16) -> u8 {
        let idx = ((addr - 0x8000) as usize) & self.prg_addr_mask;
        // SAFETY: SMB/NROM PRG ROM sizes are power-of-two and prg_addr_mask is len - 1.
        unsafe { *self.prg_rom.get_unchecked(idx) }
    }

    #[inline]
    fn cpu_read_u16(&mut self, addr: u16) -> u16 {
        let lo = self.cpu_read(addr) as u16;
        let hi = self.cpu_read(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn controller_write(&mut self, value: u8) {
        self.controller_strobe = value & 1 != 0;
        if self.controller_strobe {
            self.controller_shift = self.controller_state;
        }
    }

    #[inline]
    fn controller_read(&mut self) -> u8 {
        if self.controller_strobe {
            return 0x40 | (self.controller_state & 1);
        }
        let value = self.controller_shift & 1;
        self.controller_shift = (self.controller_shift >> 1) | 0x80;
        0x40 | value
    }

    fn oam_dma(&mut self, page: u8) {
        let base = (page as u16) << 8;
        for i in 0..256u16 {
            let value = self.cpu_read(base | i);
            let idx = self.ppu.oam_addr.wrapping_add(i as u8) as usize;
            self.ppu.oam[idx] = value;
        }
        self.extra_cycles = self.extra_cycles.wrapping_add(513);
    }

    #[inline]
    fn fetch_u8(&mut self) -> u8 {
        let value = if self.cpu.pc >= 0x8000 {
            self.prg_read(self.cpu.pc)
        } else {
            self.cpu_read(self.cpu.pc)
        };
        self.cpu.pc = self.cpu.pc.wrapping_add(1);
        value
    }

    #[inline]
    fn fetch_u16(&mut self) -> u16 {
        let lo = self.fetch_u8() as u16;
        let hi = self.fetch_u8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn zp(&mut self) -> u16 {
        self.fetch_u8() as u16
    }

    #[inline]
    fn zpx(&mut self) -> u16 {
        self.fetch_u8().wrapping_add(self.cpu.x) as u16
    }

    #[inline]
    fn zpy(&mut self) -> u16 {
        self.fetch_u8().wrapping_add(self.cpu.y) as u16
    }

    #[inline]
    fn abs(&mut self) -> u16 {
        self.fetch_u16()
    }

    #[inline]
    fn absx(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.x as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn absy(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn indx(&mut self) -> u16 {
        let ptr = self.fetch_u8().wrapping_add(self.cpu.x);
        let lo = self.cpu_read(ptr as u16) as u16;
        let hi = self.cpu_read(ptr.wrapping_add(1) as u16) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn indy(&mut self) -> (u16, bool) {
        let ptr = self.fetch_u8();
        let lo = self.cpu_read(ptr as u16) as u16;
        let hi = self.cpu_read(ptr.wrapping_add(1) as u16) as u16;
        let base = lo | (hi << 8);
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn set_flag(&mut self, flag: u8, value: bool) {
        if value {
            self.cpu.p |= flag;
        } else {
            self.cpu.p &= !flag;
        }
        self.cpu.p |= FLAG_U;
    }

    #[inline]
    fn flag(&self, flag: u8) -> bool {
        self.cpu.p & flag != 0
    }

    #[inline]
    fn set_zn(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_N);
        if value == 0 {
            p |= FLAG_Z;
        }
        if value & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn push(&mut self, value: u8) {
        let addr = 0x0100 | self.cpu.sp as u16;
        self.cpu_write(addr, value);
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    #[inline]
    fn pop(&mut self) -> u8 {
        self.cpu.sp = self.cpu.sp.wrapping_add(1);
        self.cpu_read(0x0100 | self.cpu.sp as u16)
    }

    #[inline]
    fn push_u16(&mut self, value: u16) {
        self.push((value >> 8) as u8);
        self.push(value as u8);
    }

    #[inline]
    fn pop_u16(&mut self) -> u16 {
        let lo = self.pop() as u16;
        let hi = self.pop() as u16;
        lo | (hi << 8)
    }

    fn interrupt(&mut self, vector: u16, brk: bool) {
        self.push_u16(self.cpu.pc);
        let mut p = self.cpu.p | FLAG_U;
        if brk {
            p |= FLAG_B;
        } else {
            p &= !FLAG_B;
        }
        self.push(p);
        self.set_flag(FLAG_I, true);
        self.cpu.pc = self.cpu_read_u16(vector);
    }

    fn cpu_step(&mut self) -> u16 {
        let opcode = self.fetch_u8();
        let mut cycles = match opcode {
            0x00 => {
                self.cpu.pc = self.cpu.pc.wrapping_add(1);
                self.interrupt(0xfffe, true);
                7
            }
            0x01 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.ora(v);
                6
            }
            0x05 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.ora(v);
                3
            }
            0x06 => {
                let a = self.zp();
                self.asl_mem(a);
                5
            }
            0x08 => {
                self.push(self.cpu.p | FLAG_B | FLAG_U);
                3
            }
            0x09 => {
                let v = self.fetch_u8();
                self.ora(v);
                2
            }
            0x0a => {
                self.cpu.a = self.asl(self.cpu.a);
                2
            }
            0x0d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.ora(v);
                4
            }
            0x0e => {
                let a = self.abs();
                self.asl_mem(a);
                6
            }
            0x10 => self.branch(!self.flag(FLAG_N)),
            0x11 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.ora(v);
                5 + p as u16
            }
            0x15 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.ora(v);
                4
            }
            0x16 => {
                let a = self.zpx();
                self.asl_mem(a);
                6
            }
            0x18 => {
                self.set_flag(FLAG_C, false);
                2
            }
            0x19 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.ora(v);
                4 + p as u16
            }
            0x1d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.ora(v);
                4 + p as u16
            }
            0x1e => {
                let (a, _) = self.absx();
                self.asl_mem(a);
                7
            }
            0x20 => {
                let a = self.abs();
                self.push_u16(self.cpu.pc.wrapping_sub(1));
                self.cpu.pc = a;
                6
            }
            0x21 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.and(v);
                6
            }
            0x24 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.bit(v);
                3
            }
            0x25 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.and(v);
                3
            }
            0x26 => {
                let a = self.zp();
                self.rol_mem(a);
                5
            }
            0x28 => {
                self.cpu.p = (self.pop() & !FLAG_B) | FLAG_U;
                4
            }
            0x29 => {
                let v = self.fetch_u8();
                self.and(v);
                2
            }
            0x2a => {
                self.cpu.a = self.rol(self.cpu.a);
                2
            }
            0x2c => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.bit(v);
                4
            }
            0x2d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.and(v);
                4
            }
            0x2e => {
                let a = self.abs();
                self.rol_mem(a);
                6
            }
            0x30 => self.branch(self.flag(FLAG_N)),
            0x31 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.and(v);
                5 + p as u16
            }
            0x35 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.and(v);
                4
            }
            0x36 => {
                let a = self.zpx();
                self.rol_mem(a);
                6
            }
            0x38 => {
                self.set_flag(FLAG_C, true);
                2
            }
            0x39 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.and(v);
                4 + p as u16
            }
            0x3d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.and(v);
                4 + p as u16
            }
            0x3e => {
                let (a, _) = self.absx();
                self.rol_mem(a);
                7
            }
            0x40 => {
                self.cpu.p = (self.pop() & !FLAG_B) | FLAG_U;
                self.cpu.pc = self.pop_u16();
                6
            }
            0x41 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.eor(v);
                6
            }
            0x45 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.eor(v);
                3
            }
            0x46 => {
                let a = self.zp();
                self.lsr_mem(a);
                5
            }
            0x48 => {
                self.push(self.cpu.a);
                3
            }
            0x49 => {
                let v = self.fetch_u8();
                self.eor(v);
                2
            }
            0x4a => {
                self.cpu.a = self.lsr(self.cpu.a);
                2
            }
            0x4c => {
                let a = self.abs();
                self.cpu.pc = a;
                3
            }
            0x4d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.eor(v);
                4
            }
            0x4e => {
                let a = self.abs();
                self.lsr_mem(a);
                6
            }
            0x50 => self.branch(!self.flag(FLAG_V)),
            0x51 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.eor(v);
                5 + p as u16
            }
            0x55 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.eor(v);
                4
            }
            0x56 => {
                let a = self.zpx();
                self.lsr_mem(a);
                6
            }
            0x58 => {
                self.set_flag(FLAG_I, false);
                2
            }
            0x59 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.eor(v);
                4 + p as u16
            }
            0x5d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.eor(v);
                4 + p as u16
            }
            0x5e => {
                let (a, _) = self.absx();
                self.lsr_mem(a);
                7
            }
            0x60 => {
                self.cpu.pc = self.pop_u16().wrapping_add(1);
                6
            }
            0x61 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.adc(v);
                6
            }
            0x65 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.adc(v);
                3
            }
            0x66 => {
                let a = self.zp();
                self.ror_mem(a);
                5
            }
            0x68 => {
                self.cpu.a = self.pop();
                self.set_zn(self.cpu.a);
                4
            }
            0x69 => {
                let v = self.fetch_u8();
                self.adc(v);
                2
            }
            0x6a => {
                self.cpu.a = self.ror(self.cpu.a);
                2
            }
            0x6c => {
                let ptr = self.abs();
                let lo = self.cpu_read(ptr) as u16;
                let hi_addr = (ptr & 0xff00) | ptr.wrapping_add(1) & 0x00ff;
                let hi = self.cpu_read(hi_addr) as u16;
                self.cpu.pc = lo | (hi << 8);
                5
            }
            0x6d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.adc(v);
                4
            }
            0x6e => {
                let a = self.abs();
                self.ror_mem(a);
                6
            }
            0x70 => self.branch(self.flag(FLAG_V)),
            0x71 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.adc(v);
                5 + p as u16
            }
            0x75 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.adc(v);
                4
            }
            0x76 => {
                let a = self.zpx();
                self.ror_mem(a);
                6
            }
            0x78 => {
                self.set_flag(FLAG_I, true);
                2
            }
            0x79 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.adc(v);
                4 + p as u16
            }
            0x7d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.adc(v);
                4 + p as u16
            }
            0x7e => {
                let (a, _) = self.absx();
                self.ror_mem(a);
                7
            }
            0x81 => {
                let a = self.indx();
                self.cpu_write(a, self.cpu.a);
                6
            }
            0x84 => {
                let a = self.zp();
                self.cpu_write(a, self.cpu.y);
                3
            }
            0x85 => {
                let a = self.zp();
                self.cpu_write(a, self.cpu.a);
                3
            }
            0x86 => {
                let a = self.zp();
                self.cpu_write(a, self.cpu.x);
                3
            }
            0x88 => {
                self.cpu.y = self.cpu.y.wrapping_sub(1);
                self.set_zn(self.cpu.y);
                2
            }
            0x8a => {
                self.cpu.a = self.cpu.x;
                self.set_zn(self.cpu.a);
                2
            }
            0x8c => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.y);
                4
            }
            0x8d => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.a);
                4
            }
            0x8e => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.x);
                4
            }
            0x90 => self.branch(!self.flag(FLAG_C)),
            0x91 => {
                let (a, _) = self.indy();
                self.cpu_write(a, self.cpu.a);
                6
            }
            0x94 => {
                let a = self.zpx();
                self.cpu_write(a, self.cpu.y);
                4
            }
            0x95 => {
                let a = self.zpx();
                self.cpu_write(a, self.cpu.a);
                4
            }
            0x96 => {
                let a = self.zpy();
                self.cpu_write(a, self.cpu.x);
                4
            }
            0x98 => {
                self.cpu.a = self.cpu.y;
                self.set_zn(self.cpu.a);
                2
            }
            0x99 => {
                let (a, _) = self.absy();
                self.cpu_write(a, self.cpu.a);
                5
            }
            0x9a => {
                self.cpu.sp = self.cpu.x;
                2
            }
            0x9d => {
                let (a, _) = self.absx();
                self.cpu_write(a, self.cpu.a);
                5
            }
            0xa0 => {
                let v = self.fetch_u8();
                self.cpu.y = v;
                self.set_zn(v);
                2
            }
            0xa1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                6
            }
            0xa2 => {
                let v = self.fetch_u8();
                self.cpu.x = v;
                self.set_zn(v);
                2
            }
            0xa4 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                3
            }
            0xa5 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                3
            }
            0xa6 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                3
            }
            0xa8 => {
                self.cpu.y = self.cpu.a;
                self.set_zn(self.cpu.y);
                2
            }
            0xa9 => {
                let v = self.fetch_u8();
                self.cpu.a = v;
                self.set_zn(v);
                2
            }
            0xaa => {
                self.cpu.x = self.cpu.a;
                self.set_zn(self.cpu.x);
                2
            }
            0xac => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4
            }
            0xad => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4
            }
            0xae => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4
            }
            0xb0 => self.branch(self.flag(FLAG_C)),
            0xb1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                5 + p as u16
            }
            0xb4 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4
            }
            0xb5 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4
            }
            0xb6 => {
                let a = self.zpy();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4
            }
            0xb8 => {
                self.set_flag(FLAG_V, false);
                2
            }
            0xb9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xba => {
                self.cpu.x = self.cpu.sp;
                self.set_zn(self.cpu.x);
                2
            }
            0xbc => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xbd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xbe => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xc0 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.y, v);
                2
            }
            0xc1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                6
            }
            0xc4 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.y, v);
                3
            }
            0xc5 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                3
            }
            0xc6 => {
                let a = self.zp();
                self.dec_mem(a);
                5
            }
            0xc8 => {
                self.cpu.y = self.cpu.y.wrapping_add(1);
                self.set_zn(self.cpu.y);
                2
            }
            0xc9 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.a, v);
                2
            }
            0xca => {
                self.cpu.x = self.cpu.x.wrapping_sub(1);
                self.set_zn(self.cpu.x);
                2
            }
            0xcc => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.y, v);
                4
            }
            0xcd => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4
            }
            0xce => {
                let a = self.abs();
                self.dec_mem(a);
                6
            }
            0xd0 => self.branch(!self.flag(FLAG_Z)),
            0xd1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                5 + p as u16
            }
            0xd5 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4
            }
            0xd6 => {
                let a = self.zpx();
                self.dec_mem(a);
                6
            }
            0xd8 => {
                self.set_flag(FLAG_D, false);
                2
            }
            0xd9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4 + p as u16
            }
            0xdd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4 + p as u16
            }
            0xde => {
                let (a, _) = self.absx();
                self.dec_mem(a);
                7
            }
            0xe0 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.x, v);
                2
            }
            0xe1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.sbc(v);
                6
            }
            0xe4 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.x, v);
                3
            }
            0xe5 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.sbc(v);
                3
            }
            0xe6 => {
                let a = self.zp();
                self.inc_mem(a);
                5
            }
            0xe8 => {
                self.cpu.x = self.cpu.x.wrapping_add(1);
                self.set_zn(self.cpu.x);
                2
            }
            0xe9 => {
                let v = self.fetch_u8();
                self.sbc(v);
                2
            }
            0xea => 2,
            0xec => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.x, v);
                4
            }
            0xed => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.sbc(v);
                4
            }
            0xee => {
                let a = self.abs();
                self.inc_mem(a);
                6
            }
            0xf0 => self.branch(self.flag(FLAG_Z)),
            0xf1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.sbc(v);
                5 + p as u16
            }
            0xf5 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.sbc(v);
                4
            }
            0xf6 => {
                let a = self.zpx();
                self.inc_mem(a);
                6
            }
            0xf8 => {
                self.set_flag(FLAG_D, true);
                2
            }
            0xf9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.sbc(v);
                4 + p as u16
            }
            0xfd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.sbc(v);
                4 + p as u16
            }
            0xfe => {
                let (a, _) = self.absx();
                self.inc_mem(a);
                7
            }
            _ => 2,
        };
        cycles = cycles.saturating_add(self.extra_cycles);
        self.extra_cycles = 0;
        cycles
    }

    #[inline]
    fn branch(&mut self, condition: bool) -> u16 {
        let offset = self.fetch_u8() as i8;
        if !condition {
            return 2;
        }
        let old_pc = self.cpu.pc;
        self.cpu.pc = self.cpu.pc.wrapping_add(offset as i16 as u16);
        3 + page_crossed(old_pc, self.cpu.pc) as u16
    }

    #[inline]
    fn adc(&mut self, value: u8) {
        let carry = u8::from(self.flag(FLAG_C));
        let a = self.cpu.a;
        let sum = a as u16 + value as u16 + carry as u16;
        let result = sum as u8;
        self.cpu.a = result;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_V | FLAG_Z | FLAG_N);
        if sum > 0xff {
            p |= FLAG_C;
        }
        if (!(a ^ value) & (a ^ result) & 0x80) != 0 {
            p |= FLAG_V;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn sbc(&mut self, value: u8) {
        self.adc(!value);
    }

    #[inline]
    fn cmp(&mut self, reg: u8, value: u8) {
        let result = reg.wrapping_sub(value);
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if reg >= value {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn ora(&mut self, value: u8) {
        self.cpu.a |= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn and(&mut self, value: u8) {
        self.cpu.a &= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn eor(&mut self, value: u8) {
        self.cpu.a ^= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn bit(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_V | FLAG_N);
        if self.cpu.a & value == 0 {
            p |= FLAG_Z;
        }
        p |= value & (FLAG_V | FLAG_N);
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn asl(&mut self, value: u8) -> u8 {
        let result = value << 1;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 0x80 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn lsr(&mut self, value: u8) -> u8 {
        let result = value >> 1;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 1 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn rol(&mut self, value: u8) -> u8 {
        let carry = u8::from(self.flag(FLAG_C));
        let result = (value << 1) | carry;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 0x80 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn ror(&mut self, value: u8) -> u8 {
        let carry = if self.flag(FLAG_C) { 0x80 } else { 0 };
        let result = (value >> 1) | carry;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 1 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn asl_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.asl(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn lsr_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.lsr(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn rol_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.rol(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn ror_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.ror(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn dec_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_sub(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    fn inc_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_add(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }
}

#[inline]
fn page_crossed(a: u16, b: u16) -> bool {
    (a & 0xff00) != (b & 0xff00)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestAreaAxisBin {
        start: usize,
        weights: Vec<u32>,
    }

    fn build_test_area_axis(src_len: usize, dst_len: usize) -> Vec<TestAreaAxisBin> {
        (0..dst_len)
            .map(|dst_i| {
                let start_num = dst_i * src_len;
                let end_num = (dst_i + 1) * src_len;
                let start = start_num / dst_len;
                let end = (end_num + dst_len - 1) / dst_len;
                let weights = (start..end)
                    .map(|src_i| {
                        let pixel_start = src_i * dst_len;
                        let pixel_end = (src_i + 1) * dst_len;
                        pixel_end
                            .min(end_num)
                            .saturating_sub(pixel_start.max(start_num))
                            as u32
                    })
                    .collect::<Vec<_>>();
                TestAreaAxisBin { start, weights }
            })
            .collect()
    }

    fn resize_default_area_reference(src: &[u8], dst: &mut [u8]) {
        let x_bins = build_test_area_axis(NES_WIDTH, DEFAULT_GRAY_RESIZE_WIDTH);
        let y_bins = build_test_area_axis(DEFAULT_GRAY_CROP_HEIGHT, DEFAULT_GRAY_RESIZE_HEIGHT);
        let rounding = DEFAULT_GRAY_AREA_DENOM / 2;

        for (dst_y, y_bin) in y_bins.iter().enumerate() {
            for (dst_x, x_bin) in x_bins.iter().enumerate() {
                let mut sum = 0u32;
                for (dy, &wy) in y_bin.weights.iter().enumerate() {
                    let row = (y_bin.start + dy) * NES_WIDTH;
                    for (dx, &wx) in x_bin.weights.iter().enumerate() {
                        sum += src[row + x_bin.start + dx] as u32 * wy * wx;
                    }
                }
                dst[dst_y * DEFAULT_GRAY_RESIZE_WIDTH + dst_x] =
                    ((sum + rounding) / DEFAULT_GRAY_AREA_DENOM) as u8;
            }
        }
    }

    fn set_sprite(ppu: &mut Ppu, sprite: usize, y: u8, tile: u8, attr: u8, x: u8) {
        let base = sprite * 4;
        ppu.oam[base] = y.wrapping_sub(1);
        ppu.oam[base + 1] = tile;
        ppu.oam[base + 2] = attr;
        ppu.oam[base + 3] = x;
    }

    fn make_test_ppu() -> Ppu {
        let chr_rom = (0..8192)
            .map(|idx| ((idx * 37 + idx / 11 + 23) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut ppu = Ppu::new(chr_rom, true);
        ppu.ctrl = 0x18;
        ppu.mask = 0x18;
        ppu.scroll_x_px = 37;
        ppu.scroll_x_low = 37;
        ppu.scroll_y_px = 11;
        for (idx, value) in ppu.vram.iter_mut().enumerate() {
            *value = ((idx * 13 + idx / 7 + 5) & 0xff) as u8;
        }
        for (idx, value) in ppu.palette.iter_mut().enumerate() {
            *value = ((idx * 3 + 7) & 0x3f) as u8;
        }
        ppu.oam.fill(0xff);
        set_sprite(&mut ppu, 63, 70, 3, 0x00, 40);
        set_sprite(&mut ppu, 0, 72, 5, 0x01, 42);
        set_sprite(&mut ppu, 1, 74, 7, 0x22, 44);
        set_sprite(&mut ppu, 2, 190, 9, 0xc3, 250);
        ppu
    }

    fn assert_default_area_writer_matches_scratch(ppu: &Ppu) {
        let mut native = vec![0; NES_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];
        let mut expected = vec![0; DEFAULT_GRAY_RESIZE_PIXELS];
        let mut actual = vec![0; DEFAULT_GRAY_RESIZE_PIXELS];
        let mut sprite_shadow = vec![0; NES_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];

        ppu.write_gray_frame_cropped(&mut native, DEFAULT_GRAY_CROP_TOP, DEFAULT_GRAY_CROP_HEIGHT);
        resize_default_area_reference(&native, &mut expected);
        ppu.write_gray_frame_cropped_area_84x84(&mut actual, &mut sprite_shadow);

        assert_eq!(actual, expected);
    }

    #[test]
    fn default_cropped_gray_area_writer_matches_scratch_resize() {
        let mut ppu = make_test_ppu();
        assert_default_area_writer_matches_scratch(&ppu);

        ppu.mask = 0x10;
        assert_default_area_writer_matches_scratch(&ppu);

        ppu.mask = 0x00;
        assert_default_area_writer_matches_scratch(&ppu);
    }
}
