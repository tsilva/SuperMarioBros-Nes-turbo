.PHONY: benchmark benchmark-local develop test test-python test-rust

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
RUSTFLAGS_EXT ?= -C link-arg=-undefined -C link-arg=dynamic_lookup
BENCHMARK_NUM_ENVS ?= 16
BENCHMARK_STEPS ?= 500
BENCHMARK_REPEATS ?= 3
BENCHMARK_ARGS ?=

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

benchmark: benchmark-local

benchmark-local:
	$(PYTHON) scripts/benchmark_sps.py --num-envs $(BENCHMARK_NUM_ENVS) --steps $(BENCHMARK_STEPS) --repeats $(BENCHMARK_REPEATS) $(BENCHMARK_ARGS)

test-rust:
	RUSTFLAGS="$(RUSTFLAGS_EXT)" cargo test --lib

test-python:
	$(PYTHON) -m pytest

test: test-rust test-python
