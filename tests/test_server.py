import dataclasses
import os
import time

import pytest
import torch
from fastapi.testclient import TestClient

from tinydiffusion.server import ServerConfig
from tinydiffusion.server.app import create_app
from tinydiffusion.server.service import SamplerService
from tinydiffusion.training.checkpoints import save_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.model import build_model

TINY = TrainConfig(
    image_size=8,
    base_channels=4,
    channel_mult=(1,),
    num_res_blocks=1,
    attn_resolutions=(),
    num_timesteps=20,
    sample_steps=4,
    num_samples=2,
    batch_size=4,
    num_workers=0,
    device="cpu",
)

CONDITIONAL = dataclasses.replace(TINY, num_classes=10, guidance=2.0)


@pytest.fixture
def make_config(tmp_path, wake):
    """Write a real checkpoint and point a ServerConfig at it."""

    def build(cfg=TINY, **overrides):
        diffusion = build_model(cfg)
        wake(diffusion.net)
        ema = EMA(diffusion.net, decay=0.9, warmup=0)
        optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, epoch=0, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, cfg=cfg
        )
        return ServerConfig(
            checkpoint=path, device="cpu", image_dir=tmp_path / "images", **overrides
        )

    return build


@pytest.fixture
def client(make_config):
    with TestClient(create_app(make_config())) as c:
        yield c


@pytest.fixture
def conditional_client(make_config):
    with TestClient(create_app(make_config(CONDITIONAL))) as c:
        yield c


# --- config ---------------------------------------------------------------


def test_the_default_bind_is_loopback():
    cfg = ServerConfig(checkpoint="m.pt")
    assert cfg.host == "127.0.0.1"
    assert cfg.cors_origins == ()


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_a_bad_port_is_rejected(port):
    with pytest.raises(ValueError, match="port"):
        ServerConfig(checkpoint="m.pt", port=port)


def test_a_bad_image_ceiling_is_rejected():
    with pytest.raises(ValueError, match="max_images"):
        ServerConfig(checkpoint="m.pt", max_images=0)


# --- status ---------------------------------------------------------------


def test_status_describes_the_checkpoint(client):
    body = client.get("/api/status").json()
    assert body["device"] == "cpu"
    assert body["weights"] == "ema"
    assert body["image_size"] == TINY.image_size
    assert body["num_classes"] is None
    assert body["default_steps"] == TINY.sample_steps


def test_status_reports_the_class_count_of_a_conditional_checkpoint(conditional_client):
    assert conditional_client.get("/api/status").json()["num_classes"] == 10


def test_requests_before_startup_are_refused(make_config):
    # No context manager, so lifespan never runs and nothing is loaded.
    app = create_app(make_config())
    assert TestClient(app).get("/api/status").status_code == 503


# --- sampling -------------------------------------------------------------


def test_sampling_returns_a_fetchable_png(client):
    body = client.post("/api/sample", json={"num_images": 2, "steps": 2}).json()
    assert body["num_images"] == 2
    assert body["url"] == f"/images/{body['filename']}"

    image = client.get(body["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")


def test_sampling_defaults_to_eight_images(client):
    body = client.post("/api/sample", json={"steps": 2}).json()
    assert body["num_images"] == 8


def test_a_seed_makes_the_result_reproducible(client):
    payload = {"num_images": 2, "steps": 2, "seed": 7}
    first = client.get(client.post("/api/sample", json=payload).json()["url"]).content
    second = client.get(client.post("/api/sample", json=payload).json()["url"]).content
    assert first == second


def test_different_seeds_give_different_images(client):
    a = client.get(
        client.post("/api/sample", json={"num_images": 2, "steps": 2, "seed": 1}).json()["url"]
    ).content
    b = client.get(
        client.post("/api/sample", json={"num_images": 2, "steps": 2, "seed": 2}).json()["url"]
    ).content
    assert a != b


def test_labels_are_accepted_by_a_conditional_checkpoint(conditional_client):
    r = conditional_client.post(
        "/api/sample", json={"num_images": 4, "steps": 2, "labels": [3], "guidance": 2.0}
    )
    assert r.status_code == 200


def test_a_guidance_rescale_is_accepted_by_a_conditional_checkpoint(conditional_client):
    r = conditional_client.post(
        "/api/sample",
        json={"num_images": 4, "steps": 2, "guidance": 5.0, "guidance_rescale": 0.7},
    )
    assert r.status_code == 200


def test_labels_are_refused_by_an_unconditional_checkpoint(client):
    r = client.post("/api/sample", json={"num_images": 2, "steps": 2, "labels": [1]})
    assert r.status_code == 400
    assert "unconditional" in r.json()["detail"]


def test_an_out_of_range_label_is_refused(conditional_client):
    r = conditional_client.post("/api/sample", json={"num_images": 2, "steps": 2, "labels": [10]})
    assert r.status_code == 400
    assert "10" in r.json()["detail"]


def test_too_many_images_are_refused(make_config):
    with TestClient(create_app(make_config(max_images=4))) as c:
        r = c.post("/api/sample", json={"num_images": 5, "steps": 2})
        assert r.status_code == 400
        assert "num_images" in r.json()["detail"]


def test_too_many_steps_are_refused(client):
    r = client.post("/api/sample", json={"num_images": 2, "steps": TINY.num_timesteps + 1})
    assert r.status_code == 400
    assert "steps" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"num_images": 0},  # below the schema minimum
        {"eta": 1.5},  # outside [0, 1]
        {"steps": 0},  # below the schema minimum
        {"guidance": -1.0},  # negative
        {"guidance_rescale": 1.5},  # outside [0, 1]
        {"guidance_rescale": -0.5},  # likewise
    ],
)
def test_the_schema_rejects_impossible_requests(client, payload):
    assert client.post("/api/sample", json=payload).status_code == 422


# --- image serving --------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "../config.py",
        "..%2F..%2Fpyproject.toml",
        "%2E%2E%2Fsecrets.png",
        "not-a-uuid.png",
        "0123456789abcdef0123456789abcdef.txt",
    ],
)
def test_only_names_this_server_issued_are_served(client, name):
    # A decoded traversal must not escape the image directory.
    assert client.get(f"/images/{name}").status_code == 404


def test_a_missing_image_is_a_404(client):
    assert client.get(f"/images/{'a' * 32}.png").status_code == 404


def test_the_service_rejects_a_forged_filename(make_config):
    service = SamplerService(make_config())
    with pytest.raises(ValueError, match="issued"):
        service.image_path("../../etc/passwd")


def test_the_service_resolves_its_own_filenames(make_config):
    service = SamplerService(make_config())
    path = service.sample(num_images=1, steps=2)
    assert service.image_path(path.name) == path
    assert path.parent == service.image_dir


# --- cors -----------------------------------------------------------------


def test_cors_is_off_by_default(client):
    r = client.get("/api/status", headers={"Origin": "http://evil.test"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_allows_only_the_configured_origin(make_config):
    with TestClient(create_app(make_config(cors_origins=("http://ui.test",)))) as c:
        allowed = c.get("/api/status", headers={"Origin": "http://ui.test"})
        assert allowed.headers["access-control-allow-origin"] == "http://ui.test"
        # Credentials are never allowed, so a browser cannot attach cookies.
        assert "access-control-allow-credentials" not in allowed.headers
        other = c.get("/api/status", headers={"Origin": "http://evil.test"})
        assert "access-control-allow-origin" not in other.headers


# --- retention ------------------------------------------------------------


def _stale(path, age):
    """Backdate a file so the sweep sees it as old."""
    when = time.time() - age
    os.utime(path, (when, when))


def test_retention_settings_must_be_non_negative():
    with pytest.raises(ValueError, match="image_ttl"):
        ServerConfig(checkpoint="m.pt", image_ttl=-1)
    with pytest.raises(ValueError, match="keep_images"):
        ServerConfig(checkpoint="m.pt", keep_images=-1)


def test_expired_images_are_swept(make_config):
    service = SamplerService(make_config(image_ttl=60, keep_images=0))
    old = service.sample(num_images=1, steps=2)
    _stale(old, 3600)
    fresh = service.sample(num_images=1, steps=2)

    assert not old.exists()
    assert fresh.exists()


def test_the_image_count_is_capped(make_config):
    service = SamplerService(make_config(image_ttl=0, keep_images=2))
    # Written directly rather than sampled: `sample` sweeps as it goes, and
    # same-second mtimes would leave "oldest" ambiguous.
    written = [service.image_dir / f"{index:032x}.png" for index in range(4)]
    for age, path in enumerate(reversed(written)):
        path.write_bytes(b"")
        _stale(path, age)

    assert service.prune_images() == 2
    assert [path.exists() for path in written] == [False, False, True, True]


def test_a_fresh_render_survives_its_own_sweep(make_config):
    service = SamplerService(make_config(image_ttl=0, keep_images=1))
    first = service.sample(num_images=1, steps=2)
    _stale(first, 10)
    second = service.sample(num_images=1, steps=2)

    assert second.exists()
    assert not first.exists()


def test_the_sweep_leaves_foreign_files_alone(make_config):
    service = SamplerService(make_config(image_ttl=1, keep_images=1))
    bystander = service.image_dir / "notes.txt"
    bystander.write_text("not ours")
    _stale(bystander, 3600)

    service.prune_images()
    assert bystander.exists()


def test_retention_can_be_switched_off_entirely(make_config):
    service = SamplerService(make_config(image_ttl=0, keep_images=0))
    written = [service.sample(num_images=1, steps=2) for _ in range(3)]
    for path in written:
        _stale(path, 10**6)

    assert service.prune_images() == 0
    assert all(p.exists() for p in written)


def test_status_reports_the_retention_policy(client):
    body = client.get("/api/status").json()
    assert body["image_ttl"] > 0
    assert body["keep_images"] > 0


# --- seeding --------------------------------------------------------------


def test_a_seeded_request_does_not_disturb_the_global_rng(make_config):
    """A client's seed must not reseed the process it is talking to."""
    service = SamplerService(make_config())
    torch.manual_seed(1234)
    before = torch.randn(3)

    service.sample(num_images=1, steps=2, seed=99)

    torch.manual_seed(1234)
    assert torch.equal(before, torch.randn(3))
