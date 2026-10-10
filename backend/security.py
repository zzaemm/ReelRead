"""Security helpers for the ReelRead API.

Everything here is small on purpose, so each defence is easy to read on its own:

- Cookie helpers: the anonymous user id lives in the `rr_uid` cookie as a UUID4.
- RateLimiter: an in-memory sliding-window limiter (one per route group).
- SecurityHeadersMiddleware: adds CSP and friends to EVERY response.
- OriginCheckMiddleware: CSRF defence (Origin check on POSTs, Sec-Fetch-Site on /api).
- BodySizeLimitMiddleware: 413 for any request body over its size cap.

The middlewares are written as plain ASGI middleware (a class with
`__call__(scope, receive, send)`) rather than Starlette's BaseHTTPMiddleware.
Plain ASGI is a little more code but it wraps static files, error responses and
streaming bodies uniformly, and it has no hidden buffering.
"""
from __future__ import annotations

import math
import os
import threading
import time
import uuid
from collections import deque
from typing import Callable
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# ---------------------------------------------------------------------------
# Cookie
# ---------------------------------------------------------------------------

COOKIE_NAME = "rr_uid"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year, in seconds (31536000)


def is_valid_uuid4(value: str | None) -> bool:
    """True only for a canonical, lowercase UUID4 string like the ones we issue.

    We insist on the exact canonical form (str(uuid) == value) so that variants
    such as "{...}", upper case or missing dashes are rejected. That keeps the
    value we look up in the database identical to the value we created.
    """
    if not value or len(value) != 36:
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def new_user_id() -> str:
    """A fresh random user id. uuid4 uses os.urandom, so it is unguessable."""
    return str(uuid.uuid4())


def secure_cookies_enabled() -> bool:
    """Secure cookies are only sent over HTTPS, so they are opt-in for local http dev."""
    return os.environ.get("REELREAD_SECURE_COOKIE") == "1"


def set_user_cookie(response, user_id: str) -> None:
    """Attach the identity cookie with the flags required by SPEC.md."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=user_id,
        max_age=COOKIE_MAX_AGE,
        path="/",
        httponly=True,          # JavaScript can't read it -> XSS can't steal it
        samesite="lax",         # not sent on cross-site POSTs -> CSRF defence
        secure=secure_cookies_enabled(),
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter: at most `limit` requests per `window` seconds per key.

    For each key (a user id or client IP) we keep a queue of the timestamps of
    recent allowed requests. On every new request we drop timestamps older than
    the window; if what's left is already at the limit, the request is refused.

    It lives in memory, so it resets when the server restarts and is NOT shared
    between multiple worker processes. That's fine for V1 (one uvicorn worker).
    A lock makes it safe for FastAPI's threadpool, which runs sync handlers and
    dependencies on several threads at once.
    """

    SWEEP_EVERY = 1000  # every N calls, forget keys that have gone quiet

    def __init__(self, limit: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock          # injectable so tests can fake time
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key: str) -> float | None:
        """Record one request for `key`.

        Returns None if the request is allowed, otherwise the number of seconds
        until the oldest request leaves the window (useful for Retry-After).
        Refused requests are not recorded, so a client that backs off recovers.
        """
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % self.SWEEP_EVERY == 0:
                self._sweep(now)

            timestamps = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                return max(timestamps[0] + self.window - now, 0.0)
            timestamps.append(now)
            return None

    def _sweep(self, now: float) -> None:
        """Delete keys with no recent hits so memory doesn't grow forever."""
        cutoff = now - self.window
        stale = [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]
        for k in stale:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._calls = 0


# One limiter per rule in SPEC.md ("Rate limiting").
events_limiter = RateLimiter(limit=60)          # POST /api/events, per user
feed_limiter = RateLimiter(limit=30)            # GET /api/feed, per user
quiz_limiter = RateLimiter(limit=30)            # POST /api/quiz/answer, per user
session_create_limiter = RateLimiter(limit=10)  # new users, per client IP

ALL_LIMITERS = [events_limiter, feed_limiter, quiz_limiter, session_create_limiter]


def reset_rate_limits() -> None:
    """Clear every limiter (used by tests so they don't affect each other)."""
    for limiter in ALL_LIMITERS:
        limiter.reset()


def enforce_rate_limit(limiter: RateLimiter, key: str) -> None:
    """Raise 429 Too Many Requests if `key` is over `limiter`'s limit."""
    retry_after = limiter.hit(key)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests, slow down.",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )


def client_ip(request: Request) -> str:
    """The TCP peer address. We deliberately ignore X-Forwarded-For: anyone can
    set that header, so trusting it would let a client pick its own rate-limit key.
    (Behind a real reverse proxy you'd configure uvicorn's --proxy-headers instead.)"""
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Request body size limit
# ---------------------------------------------------------------------------

MAX_EVENTS_BODY_BYTES = 16 * 1024   # /api/events carries up to 50 events
MAX_DEFAULT_BODY_BYTES = 4 * 1024   # every other route has a tiny JSON body (or none)


def body_limit_for(path: str) -> int:
    """How many request-body bytes we accept for this URL path."""
    return MAX_EVENTS_BODY_BYTES if path == "/api/events" else MAX_DEFAULT_BODY_BYTES


class BodySizeLimitMiddleware:
    """Cap EVERY request body (16 KB for /api/events, 4 KB elsewhere): 413 if larger.

    Without a cap, FastAPI happily reads a 40 MB body into memory and then tries
    to parse it as JSON. Two checks, because either alone can be dodged:

    1. Content-Length: if the client *declares* a big body, refuse straight away.
    2. Streaming count: a chunked upload has no Content-Length, so we read the
       body ourselves, counting bytes, and stop the moment it goes over.

    Because the allowed bodies are tiny, we simply buffer the (small) body here
    and then hand it to the app as if it had just arrived ("replay").
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = body_limit_for(scope.get("path", ""))
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            if not declared.strip().isdigit():
                await _reject(scope, receive, send, 400, "Invalid Content-Length.")
                return
            if int(declared) > limit:
                await _reject(scope, receive, send, 413, "Request body too large.")
                return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client hung up mid-upload; nobody to answer
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                await _reject(scope, receive, send, 413, "Request body too large.")
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> Message:
            # First call: the body we already read. Later calls: pass through
            # (the app may wait for "http.disconnect" there).
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
    response = JSONResponse({"detail": detail}, status_code=status)
    await response(scope, receive, send)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "img-src 'self' https://upload.wikimedia.org https://thumb.wikimedia.org data:; "
    "script-src 'self'; "
    "style-src 'self'; "
    "connect-src 'self'; "
    "manifest-src 'self'; "
    "worker-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
}


class SecurityHeadersMiddleware:
    """Adds SECURITY_HEADERS to every HTTP response: API, static files and errors.

    API responses also get `Cache-Control: no-store`, so per-user data is never
    kept by the browser or a proxy cache.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        is_api = scope.get("path", "").startswith("/api/")
        response_started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
                if is_api:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            # An unhandled bug: Starlette's outermost error handler would send a
            # bare 500 *without* our headers. Send our own (with headers) first,
            # then re-raise so the traceback still gets logged.
            if not response_started:
                response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
                await response(scope, receive, send_with_headers)
            raise


# ---------------------------------------------------------------------------
# CSRF: Origin check
# ---------------------------------------------------------------------------

STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_matches_host(origin: str, host: str | None, scheme: str = "http") -> bool:
    """True if the Origin header's host:port equals the request's Host header.

    Browsers omit the port when it's the default (80/443), so we fill in the
    default before comparing. Anything unparsable (including the literal
    Origin "null" sent by sandboxed pages) counts as a mismatch.
    """
    if not host:
        return False
    try:
        o = urlsplit(origin)
        if o.scheme not in DEFAULT_PORTS or not o.hostname:
            return False
        origin_port = o.port or DEFAULT_PORTS[o.scheme]

        h = urlsplit("//" + host)  # parse "example.com:8000" as a netloc
        if not h.hostname:
            return False
        host_port = h.port or DEFAULT_PORTS.get(scheme, 80)
    except ValueError:  # e.g. a non-numeric port
        return False
    return o.hostname == h.hostname and origin_port == host_port


class OriginCheckMiddleware:
    """CSRF defence: 403 for cross-site /api requests and for state-changing
    requests whose Origin doesn't match our Host.

    Browsers attach Origin to cross-site POSTs and scripts can't forge it, so a
    mismatch means another site is trying to act as our user. A *missing*
    Origin is allowed (SPEC.md): same-origin requests and sendBeacon may omit it,
    and the SameSite=Lax cookie already blocks most cross-site POSTs.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)

        # Sec-Fetch-Site is set by the browser (pages can't fake it). "cross-site"
        # means another website triggered this request, e.g. a link or <img> that
        # points at /api/feed. Even GET /api/feed changes state (served rows,
        # quiz_shown), so ANY cross-site /api request is refused. A missing
        # header (old browsers, curl) or same-origin / same-site / none is fine.
        if (scope.get("path", "").startswith("/api/")
                and headers.get("sec-fetch-site", "").lower() == "cross-site"):
            await _reject(scope, receive, send, 403, "Cross-site request blocked.")
            return

        if scope["method"] in STATE_CHANGING_METHODS:
            origin = headers.get("origin")
            if origin is not None and not origin_matches_host(
                origin, headers.get("host"), scope.get("scheme", "http")
            ):
                await _reject(scope, receive, send, 403, "Cross-origin request blocked.")
                return
        await self.app(scope, receive, send)
