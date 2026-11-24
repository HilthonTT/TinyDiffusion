"""HTTP API over a trained checkpoint.

One checkpoint per process, loaded at startup. The routes are a thin shell over
:class:`~tinydiffusion.server.service.SamplerService`: validate the request,
run the sampler off the event loop, hand back a URL.

FastAPI is an optional dependency — install the ``server`` extra — so this
module is only importable when it is present. Everything else in the package
works without it.
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tinydiffusion.server.config import MAX_REQUEST_BYTES, ServerConfig
from tinydiffusion.server.service import SamplerService
from tinydiffusion.version import __version__

logger = logging.getLogger("tinydiffusion.server")


class BodySizeLimit:
    """Refuse request bodies larger than a fixed number of bytes.

    Neither uvicorn nor FastAPI bounds a request body: it is read into memory
    whole and only then validated, so a schema that rejects a huge label list
    still lets the huge label list be uploaded. This sits in front of the
    routes and stops reading at the limit instead.

    Two checks, because a client need not declare a length. A declared
    ``Content-Length`` over the limit is refused before a byte of body is read;
    a chunked body is counted as it arrives and refused once it crosses.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        """Wrap `app`.

        Args:
            app: the application to protect.
            max_bytes: largest body accepted.
        """
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request through, or answer 413 if its body is too large."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            response = JSONResponse({"detail": "request body too large"}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises an HTTPException from body reading
                    # rather than folding it into a generic 400, so this
                    # reaches the client as the 413 it is.
                    raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, limited, send)


class SampleRequest(BaseModel):
    """A request for a grid of generated images.

    Attributes:
        num_images: how many images to generate.
        labels: classes to generate, cycled over the grid. Conditional
            checkpoints only; omit for one image per class.
        guidance: classifier-free guidance scale, or null for the checkpoint's.
        guidance_rescale: how much of the scale inflation guidance causes to
            correct, in [0, 1], or null for the checkpoint's. 0.7 is the
            published value; worth setting whenever guidance is above about 3.
        steps: DDIM steps, or null for the checkpoint's.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
        seed: seed applied before sampling, or null to leave the RNG alone.
    """

    num_images: int = Field(default=8, ge=1)
    labels: list[int] | None = None
    # Python's JSON parser accepts the non-standard ``Infinity``; an infinite
    # scale passes ``ge`` and spends a whole render producing NaN images.
    guidance: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    guidance_rescale: float | None = Field(default=None, ge=0.0, le=1.0)
    steps: int | None = Field(default=None, ge=1)
    eta: float = Field(default=0.0, ge=0.0, le=1.0)
    # The range torch.Generator.manual_seed accepts; past it, it overflows.
    seed: int | None = Field(default=None, ge=-(2**63), le=2**64 - 1)


class SampleResponse(BaseModel):
    """Where to fetch the grid that was generated.

    Attributes:
        url: path to the PNG, relative to the server root.
        filename: the PNG's name on its own.
        num_images: how many images the grid holds.
    """

    url: str
    filename: str
    num_images: int


def create_app(config: ServerConfig) -> FastAPI:
    """Build the application around one checkpoint.

    A factory rather than a module-level app: the checkpoint has to be chosen
    before anything is loaded, and a module-level instance would load it at
    import time, where a failure is a stack trace from an import rather than a
    message from a command.

    Args:
        config: server settings, including the checkpoint to serve.

    Returns:
        The configured application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        logger.info("loading %s", config.checkpoint)
        app.state.service = await run_in_threadpool(SamplerService, config)
        logger.info("ready on %s", app.state.service.device)
        yield
        service: SamplerService = app.state.service
        app.state.service = None
        service.close()

    inflight = anyio.Semaphore(config.max_inflight)

    app = FastAPI(
        title="TinyDiffusion",
        version=__version__,
        summary="Sample images from a trained TinyDiffusion checkpoint.",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimit)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Report what was wrong with a request, without echoing it back.

        FastAPI's own handler returns each offending value under ``input``.
        Python's JSON parser accepts ``Infinity`` and ``NaN``, which strict
        JSON cannot encode, so a request carrying one made the 422 itself fail
        to serialise and the caller got a 500 instead. Where and why is all a
        caller needs; they already have the value.
        """
        detail = [
            {"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse({"detail": detail}, status_code=422)

    if config.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def get_service() -> SamplerService:
        """Fetch the loaded service, or fail the request cleanly.

        Returns:
            The service loaded at startup.

        Raises:
            HTTPException: 503 if startup has not finished.
        """
        service: SamplerService | None = getattr(app.state, "service", None)
        if service is None:
            raise HTTPException(503, "checkpoint is not loaded")
        return service

    @app.get("/api/status")
    async def status(service: Annotated[SamplerService, Depends(get_service)]) -> dict[str, object]:
        """Describe the checkpoint being served."""
        return service.status()

    @app.post("/api/sample")
    async def sample(
        request: SampleRequest,
        service: Annotated[SamplerService, Depends(get_service)],
    ) -> SampleResponse:
        """Generate a grid of images.

        Raises:
            HTTPException: 400 if the request does not fit the checkpoint, or
                503 if too many callers are already waiting to be served.
        """
        try:
            plan = service.plan(
                num_images=request.num_images,
                labels=request.labels,
                guidance=request.guidance,
                guidance_rescale=request.guidance_rescale,
                steps=request.steps,
                eta=request.eta,
                seed=request.seed,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        try:
            inflight.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(
                503,
                "the server is already sampling as many requests as it accepts at once; "
                "retry shortly",
                headers={"Retry-After": "5"},
            ) from None

        try:
            path = await run_in_threadpool(service.render, plan)
        finally:
            inflight.release()

        return SampleResponse(
            url=f"/images/{path.name}", filename=path.name, num_images=request.num_images
        )

    @app.get("/images/{filename}", response_class=FileResponse)
    async def image(
        service: Annotated[SamplerService, Depends(get_service)],
        filename: Annotated[str, PathParam(description="Name returned by /api/sample.")],
    ) -> FileResponse:
        """Serve a generated PNG.

        Raises:
            HTTPException: 404 if the name is not one this server issued, or
                the file is gone.
        """
        try:
            path = service.image_path(filename)
        except ValueError as exc:
            raise HTTPException(404, "image not found") from exc
        # Every file this server writes is a regular file. A symlink in the
        # image directory was planted by someone else, and following it would
        # serve whatever it points at with this process's permissions.
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "image not found")
        return FileResponse(path, media_type="image/png")

    return app


def serve(config: ServerConfig) -> None:
    """Run the application under uvicorn until interrupted.

    Args:
        config: server settings.

    Raises:
        ImportError: if the ``server`` extra is not installed.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the server needs the 'server' extra: pip install 'tinydiffusion[server]'"
        ) from exc

    uvicorn.run(create_app(config), host=config.host, port=config.port)
