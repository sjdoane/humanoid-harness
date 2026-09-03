"""Serve package-owned UI files and read-only project evidence."""

from __future__ import annotations

import ipaddress
import json
import threading
import webbrowser
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlsplit

from ..research import KnowledgeIndexError, index_stats, query_index
from ..research.knowledge import DEFAULT_DATABASE
from ..status import program_status
from .local_evidence import LocalEvidenceError, local_exploration_status, local_media

STATIC_DIR = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class EvidenceServer(ThreadingHTTPServer):
    """HTTP server with explicit project and graph paths."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        project_root: Path,
        database: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.database = database.resolve()
        super().__init__(server_address, handler)
        self.local_media_enabled = ipaddress.ip_address(self.server_address[0]).is_loopback


class EvidenceRequestHandler(BaseHTTPRequestHandler):
    server: EvidenceServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _headers(
        self,
        status: HTTPStatus,
        content_type: str,
        length: int,
        *,
        extra: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; "
            "img-src 'self'; media-src 'self'; frame-ancestors 'none'",
        )
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        item = STATIC_FILES.get(path)
        if item is None:
            self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        filename, content_type = item
        body = (STATIC_DIR / filename).read_bytes()
        self._headers(HTTPStatus.OK, content_type, len(body))
        self.wfile.write(body)

    def _has_valid_host(self) -> bool:
        value = self.headers.get("Host")
        if not value or any(ord(character) < 33 for character in value):
            return False
        try:
            parsed = urlsplit(f"//{value}")
            port = parsed.port
        except ValueError:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.path or parsed.query or parsed.fragment:
            return False
        if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            return False
        expected_port = int(self.server.server_address[1])
        return port == expected_port or (port is None and expected_port == 80)

    @staticmethod
    def _byte_range(value: str, size: int) -> tuple[int, int]:
        if not value.startswith("bytes=") or "," in value:
            raise ValueError("invalid byte range")
        bounds = value.removeprefix("bytes=").split("-", maxsplit=1)
        if len(bounds) != 2:
            raise ValueError("invalid byte range")
        start_text, end_text = bounds
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError("invalid byte range")
            start = max(0, size - suffix)
            return start, size - 1
        start = int(start_text)
        end = size - 1 if not end_text else int(end_text)
        if start < 0 or start >= size or end < start:
            raise ValueError("invalid byte range")
        return start, min(end, size - 1)

    def _local_media(self, route_name: str, *, head_only: bool = False) -> None:
        if not self.server.local_media_enabled:
            self._json({"error": "local_evidence_requires_loopback"}, HTTPStatus.NOT_FOUND)
            return
        try:
            source, content_type = local_media(self.server.project_root, route_name)
            size = len(source)
        except LocalEvidenceError:
            self._json({"error": "local_evidence_unavailable"}, HTTPStatus.NOT_FOUND)
            return

        range_header = self.headers.get("Range")
        if range_header is None:
            start, end = 0, size - 1
            status = HTTPStatus.OK
            extra = {"Accept-Ranges": "bytes"}
        else:
            try:
                start, end = self._byte_range(range_header, size)
            except (TypeError, ValueError):
                self._headers(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    "application/json; charset=utf-8",
                    0,
                    extra={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
                )
                return
            status = HTTPStatus.PARTIAL_CONTENT
            extra = {
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{size}",
            }

        length = end - start + 1
        self._headers(status, content_type, length, extra=extra)
        if head_only:
            return
        for offset in range(start, end + 1, 64 * 1024):
            self.wfile.write(source[offset : min(offset + 64 * 1024, end + 1)])

    def do_GET(self) -> None:
        if not self._has_valid_host():
            self._json({"error": "invalid_host"}, HTTPStatus.MISDIRECTED_REQUEST)
            return
        request = urlparse(self.path)
        if request.path == "/health":
            self._json({"status": "ok", "authority": "read_only"})
            return
        if request.path == "/api/status":
            self._json(
                program_status(
                    project_root=self.server.project_root,
                    database=self.server.database,
                )
            )
            return
        if request.path == "/api/research/stats":
            try:
                result = index_stats(self.server.database)
            except KnowledgeIndexError as exc:
                self._json({"state": "not_built", "detail": str(exc)})
            else:
                self._json({"state": "built", **result})
            return
        if request.path == "/api/research/query":
            values = parse_qs(request.query)
            query = values.get("q", [""])[0]
            try:
                result = query_index(query, self.server.database, limit=12)
            except KnowledgeIndexError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else:
                self._json({"query": query, "results": result})
            return
        if request.path == "/api/exploration/latest":
            if not self.server.local_media_enabled:
                self._json(
                    {
                        "state": "unavailable",
                        "authority": "local_exploration_only",
                        "detail": "local evidence is available only on a loopback server",
                    }
                )
            else:
                self._json(local_exploration_status(self.server.project_root))
            return
        if request.path.startswith("/local-evidence/"):
            self._local_media(request.path.removeprefix("/local-evidence/"))
            return
        self._static(request.path)

    def do_HEAD(self) -> None:
        if not self._has_valid_host():
            self._headers(
                HTTPStatus.MISDIRECTED_REQUEST,
                "application/json; charset=utf-8",
                0,
            )
            return
        request = urlparse(self.path)
        if request.path.startswith("/local-evidence/"):
            self._local_media(
                request.path.removeprefix("/local-evidence/"),
                head_only=True,
            )
            return
        self._headers(HTTPStatus.METHOD_NOT_ALLOWED, "text/plain; charset=utf-8", 0)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    project_root: Path | None = None,
    database: Path = DEFAULT_DATABASE,
) -> EvidenceServer:
    if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
        raise ValueError("port must be in [0, 65535]")
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("the evidence UI may bind only to a loopback host")
    return EvidenceServer(
        (host, port),
        EvidenceRequestHandler,
        project_root=project_root or Path.cwd(),
        database=database,
    )


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    project_root: Path | None = None,
    database: Path = DEFAULT_DATABASE,
    open_browser: bool = True,
) -> None:
    server = create_server(
        host=host,
        port=port,
        project_root=project_root,
        database=database,
    )
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}"
    print(f"Humanoid Harness evidence UI: {url}")
    print("Authority: read-only derived view; canonical files and receipts remain authoritative.")
    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
