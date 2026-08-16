# Contributing to TinyDiffusion

Thanks for your interest in improving TinyDiffusion.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management, and targets **Python 3.14**.

```bash
git clone https://github.com/HilthonTT/TinyDiffusion.git
cd TinyDiffusion

uv python install 3.14
uv sync --all-extras --dev
uv run pre-commit install
```

`uv sync` installs a CUDA build of PyTorch on Windows and Linux, and a CPU one
on macOS. For a CPU-only environment anywhere — which is what CI uses — ignore
the PyTorch index pinned in `pyproject.toml`:

```bash
uv sync --all-extras --dev --no-sources
```

See [docs/INSTALL.md](docs/INSTALL.md) for the details, and for checking that
the wheel supports your card.

## Everyday commands

| Task              | Command                                     |
| ----------------- | ------------------------------------------- |
| Run tests         | `uv run pytest`                             |
| Fast tests only   | `uv run pytest -m "not slow and not gpu"`   |
| Coverage          | `uv run pytest --cov`                       |
| Lint              | `uv run ruff check .`                       |
| Autofix + format  | `uv run ruff check --fix . && uv run ruff format .` |
| Type-check        | `uv run mypy`                               |
| All hooks         | `uv run pre-commit run --all-files`         |

CI runs exactly these checks, so a clean `pre-commit run --all-files` plus
`uv run pytest` means CI will be green.

## Branches and commits

- Branch off `main`: `feat/short-description`, `fix/short-description`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `ci:`, `chore:`.
  Use `!` or a `BREAKING CHANGE:` footer for breaking changes.
- Keep PRs focused. A refactor and a behaviour change belong in separate PRs.

## Code standards

- **Type annotations are required** on all public functions; `mypy` runs in strict mode.
- **Docstrings** follow the Google convention on every public module member.
- **Line length is 100**; ruff's formatter is authoritative — don't hand-format.
- Never commit checkpoints, datasets or generated samples. `.gitignore` covers the
  usual paths, and a pre-commit hook rejects files over 1 MB.

## Testing

- Every bug fix gets a regression test.
- Mark anything over a few seconds with `@pytest.mark.slow`, and anything needing
  CUDA with `@pytest.mark.gpu`. CI deselects `gpu`.
- Tests must be deterministic — seed via the `seed_everything` helper or the
  autouse fixture in `tests/conftest.py`.
- For model code, prefer shape/invariant assertions (does the noise schedule stay
  in `[0, 1]`? does the reverse step preserve shape?) over golden-value tests
  that break on every kernel change.

## Releasing

1. Update `CHANGELOG.md` and bump `version` in `pyproject.toml`.
2. Tag: `git tag v0.2.0 && git push origin v0.2.0`.
3. The `Release` workflow builds and publishes to PyPI via trusted publishing.
