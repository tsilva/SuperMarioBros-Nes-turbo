.PHONY: develop test test-full test-python test-python-stress test-rust

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
RUSTFLAGS_EXT ?= -C link-arg=-undefined -C link-arg=dynamic_lookup

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

test-rust:
	RUSTFLAGS="$(RUSTFLAGS_EXT)" cargo test --lib

test-python:
	$(PYTHON) -m pytest

test-python-stress:
	SUPERMARIOBROSNES_RETRO_STRESS=1 $(PYTHON) -m pytest

test: test-rust test-python

test-full: develop test-rust test-python-stress
