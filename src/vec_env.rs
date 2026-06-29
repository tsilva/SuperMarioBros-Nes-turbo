use crate::cartridge::Cartridge;
use crate::emulator::{
    MarioAction, NesEmulator, FRAME_PIXELS_GRAY, FRAME_PIXELS_RGB, NES_HEIGHT, NES_WIDTH,
    RGB_CHANNELS,
};
use rayon::prelude::*;

const PARALLEL_ENV_THRESHOLD: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct VecEnvConfig {
    pub num_envs: usize,
    pub frame_skip: usize,
    pub grayscale: bool,
    pub frame_stack: usize,
    pub terminate_on_flag: bool,
}

impl VecEnvConfig {
    pub fn channels(&self) -> usize {
        if self.grayscale {
            self.frame_stack
        } else {
            self.frame_stack * RGB_CHANNELS
        }
    }

    pub fn obs_len_per_env(&self) -> usize {
        self.channels() * NES_HEIGHT * NES_WIDTH
    }
}

pub struct MarioVecEnv {
    config: VecEnvConfig,
    envs: Vec<NesEmulator>,
}

impl MarioVecEnv {
    pub fn new(cart: Cartridge, config: VecEnvConfig) -> Self {
        let envs = (0..config.num_envs)
            .map(|_| NesEmulator::new_with_options(cart.clone(), config.terminate_on_flag))
            .collect::<Vec<_>>();
        Self { config, envs }
    }

    pub fn config(&self) -> VecEnvConfig {
        self.config
    }

    pub fn reset_into(&mut self, obs: &mut [u8]) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            self.envs
                .par_iter_mut()
                .zip(obs.par_chunks_mut(obs_stride))
                .for_each(|(env, obs_chunk)| {
                    env.reset();
                    write_reset_stack(config, env, obs_chunk);
                });
        } else {
            for env_idx in 0..config.num_envs {
                self.envs[env_idx].reset();
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                write_reset_stack(config, &self.envs[env_idx], &mut obs[start..end]);
            }
        }
    }

    pub fn step_into(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        lives: &mut [u8],
    ) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            self.envs
                .par_iter_mut()
                .zip(actions.par_iter())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(lives.par_iter_mut())
                .for_each(
                    |(
                        (
                            (
                                ((((env, action), obs_chunk), reward_out), terminated_out),
                                truncated_out,
                            ),
                            x_out,
                        ),
                        lives_out,
                    )| {
                        step_one(
                            config,
                            env,
                            *action,
                            obs_chunk,
                            reward_out,
                            terminated_out,
                            truncated_out,
                            x_out,
                            lives_out,
                        );
                    },
                );
        } else {
            for env_idx in 0..config.num_envs {
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                step_one(
                    config,
                    &mut self.envs[env_idx],
                    actions[env_idx],
                    &mut obs[start..end],
                    &mut rewards[env_idx],
                    &mut terminated[env_idx],
                    &mut truncated[env_idx],
                    &mut x_pos[env_idx],
                    &mut lives[env_idx],
                );
            }
        }
    }
}

fn step_one(
    config: VecEnvConfig,
    env: &mut NesEmulator,
    action_id: u8,
    obs_chunk: &mut [u8],
    reward_out: &mut f32,
    terminated_out: &mut bool,
    truncated_out: &mut bool,
    x_out: &mut u16,
    lives_out: &mut u8,
) {
    let action = MarioAction::from_u8(action_id);
    let mut reward = 0.0;
    for _ in 0..config.frame_skip {
        reward += env.step_frame(action);
        if env.is_done() {
            break;
        }
    }
    shift_stack_left(config, obs_chunk);
    write_current_frame_to_last_stack_slot(config, env, obs_chunk);

    *reward_out = reward;
    *terminated_out = env.is_done();
    *truncated_out = false;
    *x_out = env.x_pos();
    *lives_out = env.lives();
}

fn write_reset_stack(config: VecEnvConfig, env: &NesEmulator, obs_chunk: &mut [u8]) {
    let frame_len = frame_len(config);
    for stack_i in 0..config.frame_stack {
        let dst_start = stack_i * frame_len;
        let dst_end = dst_start + frame_len;
        write_current_frame(config, env, &mut obs_chunk[dst_start..dst_end]);
    }
}

fn shift_stack_left(config: VecEnvConfig, obs_chunk: &mut [u8]) {
    if config.frame_stack <= 1 {
        return;
    }

    let frame_len = frame_len(config);
    let move_len = (config.frame_stack - 1) * frame_len;
    obs_chunk.copy_within(frame_len..frame_len + move_len, 0);
}

fn write_current_frame_to_last_stack_slot(
    config: VecEnvConfig,
    env: &NesEmulator,
    obs_chunk: &mut [u8],
) {
    let frame_len = frame_len(config);
    let dst_start = (config.frame_stack - 1) * frame_len;
    let dst_end = dst_start + frame_len;
    write_current_frame(config, env, &mut obs_chunk[dst_start..dst_end]);
}

fn write_current_frame(config: VecEnvConfig, env: &NesEmulator, dst: &mut [u8]) {
    if config.grayscale {
        env.write_gray_frame(dst);
    } else {
        env.write_rgb_frame(dst);
    }
}

#[inline]
fn frame_len(config: VecEnvConfig) -> usize {
    if config.grayscale {
        FRAME_PIXELS_GRAY
    } else {
        FRAME_PIXELS_RGB
    }
}
