.PHONY: benchmark benchmark-local develop release test test-python test-rust

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
RUSTFLAGS_EXT ?= -C link-arg=-undefined -C link-arg=dynamic_lookup
else
RUSTFLAGS_EXT ?=
endif
BENCHMARK_NUM_ENVS ?= 16
BENCHMARK_STEPS ?= 500
BENCHMARK_REPEATS ?= 3
BENCHMARK_ARGS ?=

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

benchmark: benchmark-local

benchmark-local:
	$(PYTHON) scripts/benchmark_sps.py --num-envs $(BENCHMARK_NUM_ENVS) --steps $(BENCHMARK_STEPS) --repeats $(BENCHMARK_REPEATS) $(BENCHMARK_ARGS)

release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --extra dev --group dev
	scripts/release.py

test-rust:
	RUSTFLAGS="$(RUSTFLAGS_EXT)" cargo test --lib

test-python:
	$(PYTHON) -m pytest

test: test-rust test-python
