# Security Policy

## Supported versions

TinyDiffusion is pre-1.0. Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/HilthonTT/TinyDiffusion/security/advisories/new)
rather than a public issue. Expect an initial response within 7 days.

## Model checkpoints

`torch.load` executes arbitrary code when unpickling. Only load checkpoints you
trust, and prefer `weights_only=True` (the default since PyTorch 2.6) or the
`safetensors` format for anything downloaded from the internet. Reports of this
project loading untrusted checkpoints unsafely are in scope.
