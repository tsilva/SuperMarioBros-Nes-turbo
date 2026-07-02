---
name: build-release
description: Build, validate, and prepare SuperMarioBros-Nes-turbo PyPI release wheels. Use when the user says /build-release, asks to build release wheels, asks to tag a version, or asks for macOS and Linux supermariobrosnes-turbo wheels. This skill does not upload to PyPI.
---

# Build Release

Use this skill to cut a `supermariobrosnes-turbo` release tag and build
publish-ready wheels for macOS arm64 and Linux x86_64. This repo is not a fork,
so versioning is owned here: use normal project versions from `pyproject.toml`
and `Cargo.toml`, not upstream-aligned `.postN` versions unless the user
explicitly asks for one.

Keep fragile release mechanics in `.codex/skills/build-release/scripts/release_build.py`.
Run that helper instead of retyping long shell workflows. Do not upload to PyPI;
the final output should print the exact upload command for the user to run or
approve separately.

Do not run this skill unless the current branch is fully clean and synchronized:
all changes committed, no untracked files, upstream configured, remote state
fetched, and no commits ahead of or behind upstream. Stop before changing
versions, creating tags, or running release-build commands if any part of this
gate fails. Do not commit, push, pull, clean files, create branches, or switch
branches unless the user explicitly asks for that.

## Flow

1. Verify the release gate:

```bash
git status --short --branch
```

The output must contain only the branch line. If there are modified, deleted, or
untracked files, stop and tell the user the tree must be committed or cleaned
before building a release.

```bash
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

This must print an upstream branch. If it fails, stop and ask the user to set an
upstream before building a release.

```bash
git fetch --prune
git rev-list --left-right --count HEAD...@{u}
```

The rev-list output must be `0 0` (Git usually separates them with a tab). If
the first number is nonzero, local commits have not been pushed; stop. If the
second number is nonzero, the branch has not been pulled; stop. If both are
nonzero, the branch has diverged; stop.

2. Confirm package identity and version consistency:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py check-version
```

This must report package name `supermariobrosnes-turbo` and matching versions
across `pyproject.toml`, `Cargo.toml`, and this repo's package entry in
`Cargo.lock`.

3. Confirm release tooling before mutating versions:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py check-tools
```

This must find `maturin`, `cibuildwheel`, and `twine` in the release Python
environment, plus `cargo` and `docker` for native and Linux builds. If tooling
is missing, stop and ask whether to install or add release tooling to the repo's
dev dependencies; do not fetch packages implicitly during a release.

4. Choose the target version.

If the user gave an exact version, use it. If the user said "next patch",
"next minor", or "next major", compute it with:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py bump-version --part patch
uv run python .codex/skills/build-release/scripts/release_build.py bump-version --part minor
uv run python .codex/skills/build-release/scripts/release_build.py bump-version --part major
```

If the user only says "next version" and no project context resolves the
ambiguity, default to the next patch version and state that choice before
mutating files.

5. Check PyPI has no existing file for the target version:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py check-pypi --version <target-version>
```

If the version already exists on PyPI, stop. PyPI files are immutable.

6. Bump repo-local versions:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py bump-version --to <target-version> --write
uv lock --offline
cargo generate-lockfile
uv run python .codex/skills/build-release/scripts/release_build.py check-version --version <target-version>
```

This updates `pyproject.toml` and `Cargo.toml`; `uv lock --offline` and
`cargo generate-lockfile` refresh generated lock metadata. If `uv lock
--offline` cannot refresh from local cache, stop and report it instead of
fetching fresh packages without approval.

7. Run focused release tests before tagging:

```bash
uv run maturin develop --release
make test
uv run python scripts/smoke_smb.py
```

If the ROM smoke cannot run because the local ROM is absent, say so explicitly
and continue only if the user accepts that gap.

8. Commit the version bump only if the user has asked to cut the release in the
repo, then tag the version commit:

```bash
git add pyproject.toml Cargo.toml Cargo.lock uv.lock
git commit -m "Release v<target-version>"
git tag v<target-version> HEAD
git rev-parse v<target-version>^{commit}
git rev-parse HEAD
```

The two commit hashes must match. If the tag already exists, stop; never move or
overwrite a release tag. If the user asked only to prepare a build plan, stop
before this step.

9. Create clean source copies under `/private/tmp`:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py prepare-sources --version <target-version>
```

Use the JSON output paths for the platform builds. The helper excludes stale
build outputs, local venv/cache state, wheelhouses, pycache files, benchmark
artifacts, and compiled extension artifacts.

10. Print platform build commands:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py build-commands \
  --version <target-version> \
  --macos-src /private/tmp/<build-root>/macos-src \
  --linux-src /private/tmp/<build-root>/linux-src-clean
```

Run the macOS command locally. Run the Linux `cibuildwheel` command from the
clean Linux source copy; Docker access may be required. The Rust extension uses
`abi3-py39`, so expect one `cp39-abi3` wheel per platform rather than one wheel
per Python minor version.

11. Smoke-test each built wheel from outside the checkout:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py smoke-wheel \
  wheelhouse-v<version>-macos/<macos-wheel>.whl

uv run python .codex/skills/build-release/scripts/release_build.py smoke-wheel \
  wheelhouse-v<version>-linux/<linux-wheel>.whl
```

The smoke must import from a temp install target, not from the checkout, so the
source tree cannot shadow the built wheel.

12. Run final validation:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py final-check --version <target-version>
```

This audits wheel contents, rejects pycache/source-tree artifacts and ROM
payloads, runs `twine check`, prints SHA256 hashes, and prints the exact
`twine upload` command. Do not run the upload command unless the user explicitly
asks for publishing.

## Final Response

Report the target version, release tag, wheel paths, SHA256 hashes, validation
results, notable warnings or skipped checks, and the concrete upload command
printed by `final-check`.
