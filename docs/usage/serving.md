# Serving a checkpoint over HTTP

Putting a trained checkpoint behind a JSON API.

Part of [Usage](../../USAGE.md).

## Serving a checkpoint over HTTP

`serve` puts a checkpoint behind a small JSON API, so something other than a
shell can ask it for digits. It needs the `server` extra:

```bash
uv sync --extra server            # or: pip install 'tinydiffusion[server]'
./scripts/run.sh serve --checkpoint checkpoints/last.pt
```

```
serving checkpoints/last.pt on http://127.0.0.1:8000
INFO:     Application startup complete.
```

The checkpoint is loaded once at startup, not per request. Interactive API docs
are at `/docs`, and the schema at `/openapi.json`.

**`POST /api/sample`** generates a grid and returns where to fetch it. Every
field is optional except that the defaults come from the checkpoint:

```bash
curl -X POST localhost:8000/api/sample -H 'content-type: application/json' \
  -d '{"num_images": 8, "labels": [3], "guidance": 2.0, "steps": 50, "seed": 0}'
```

```json
{"url": "/images/d3d0e07831c3442197753ea2d7f367f9.png",
 "filename": "d3d0e07831c3442197753ea2d7f367f9.png",
 "num_images": 8}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `num_images` | 8 | Images in the grid, up to `--max-images` |
| `labels` | one per class | Classes to generate. Conditional checkpoints only |
| `guidance` | the checkpoint's | Classifier-free guidance scale |
| `guidance_rescale` | the checkpoint's | Guidance rescale factor, in [0, 1] |
| `steps` | the checkpoint's | DDIM steps |
| `eta` | 0.0 | 0 is DDIM, 1 is ancestral DDPM |
| `seed` | null | Fixes the sample; the same seed returns the same image |

`seed` is request-local: it seeds a generator used for that one sample and
nothing else. It does not reseed the server, so one caller's seed cannot reach
into another caller's images or outlive the request that asked for it.

**`GET /images/{filename}`** serves the PNG. **`GET /api/status`** reports what
is loaded — device, image size, class count, and the defaults above — which is
also how a client learns whether it may send `labels`.

A request that does not fit the checkpoint comes back as a 400 with the reason
(`labels` against an unconditional model, a class that does not exist, more
images than the ceiling); a malformed one is a 422 from the schema.

The server draws one request at a time — they share one network on one device —
so `--max-inflight` bounds how many callers may be waiting on that. Past the
limit the answer is a 503 with a `Retry-After` header rather than a place in the
queue: the wait is otherwise unbounded, and a caller who knows the server is
busy can decide what to do about it. A request that does not fit the checkpoint
is still answered with its 400 while the server is busy, since checking one
costs nothing.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to serve |
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | 8000 | Port to bind |
| `--max-images` | 64 | Ceiling on `num_images` per request |
| `--max-inflight` | 4 | Sampling requests accepted at once; the rest get a 503 |
| `--image-dir` | a temp dir | Where PNGs are written |
| `--image-ttl` | 3600 | Seconds a PNG is kept before it is swept. 0 keeps them forever |
| `--keep-images` | 256 | PNGs retained regardless of age, newest first. 0 for no cap |
| `--cors-origin` | none | Origin allowed to call the API from a browser. Repeatable |
| `--no-ema` | off | Serve the raw weights instead of the EMA |
| `--device` | auto | `cuda`, `cpu`, … |
| `--precision` | `fp32` | `fp32`, `tf32`, `fp16` or `bf16`; see [Half precision](sampling.md#half-precision) |

Two things to know before exposing it:

- **There is no authentication**, which is why the default bind is loopback
  rather than `0.0.0.0`. Generating an image is seconds of GPU time on request,
  so an open port is a denial-of-service invitation. Widen it only behind
  something that does authenticate.
- **Requests are serialised.** One checkpoint on one device, one chain at a
  time; concurrent callers queue rather than fighting over VRAM. Throughput
  comes from `num_images` in a single request, not from parallel requests.
- **Rendered PNGs are swept.** Every request writes a file, and nothing else
  deletes them, so the image directory is bounded by age (`--image-ttl`) and by
  count (`--keep-images`). The sweep only ever touches names the server itself
  issued, so pointing `--image-dir` at a directory holding anything else is
  safe. Turn both to 0 to keep everything — reasonable for a short-lived local
  server, a slow disk leak for anything longer.
