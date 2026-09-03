"""Serve package-owned UI files and read-only project evidence."""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..research import KnowledgeIndexError, index_stats, query_index
from ..research.knowledge import DEFAULT_DATABASE
from ..status import program_status

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


class EvidenceRequestHandler(BaseHTTPRequestHandler):
    server: EvidenceServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'",
        )
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

    def do_GET(self) -> None:
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
        self._static(request.path)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    project_root: Path | None = None,
    database: Path = DEFAULT_DATABASE,
) -> EvidenceServer:
    if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
        raise ValueError("port must be in [0, 65535]")
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
