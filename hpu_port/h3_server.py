"""h3_server.py — sdcpp-compatible HTTP front-end for the H3 t2va engine
(v100, goal 3: stable diffusion.cpp-API-compatible server).

Runs in the PARENT process (no HPU devices). Requests flow to the resident
CP worker pool through a file spool:

    <server_dir>/spool/req_{seq:06d}_{id}.json     queued requests
    <server_dir>/spool/{id}.cancel                 cancel markers
    <server_dir>/jobs/{id}.json                    job status (rank0 writes)
    <server_dir>/serve_sync/current.json           claim broadcast (rank0)
    <server_dir>/SHUTDOWN                          clean shutdown switch

API (subset of stable-diffusion.cpp examples/server/api.md):
    POST /sdcpp/v1/vid_gen            -> 202 {id, kind, status, created, poll_url}
    GET  /sdcpp/v1/jobs/{id}          -> job status; completed carries result.b64_json
    POST /sdcpp/v1/jobs/{id}/cancel   -> 202 (cancel before claim)
    GET  /sdcpp/v1/capabilities       -> supported modes/defaults
    GET  /health                      -> liveness

Deviations from the sdcpp contract (documented, deliberate):
- output_format: only `mp4` (H.264+AAC via ffmpeg). webm/webp/avi -> 400.
- video_frames/fps/width/height are SERVER-WIDE (fixed at server start): a
  per-request duration would change the DiT graph shape and recompile the
  giant graph per request. The fixed values are echoed in capabilities and in
  every job record. prompt/steps/seed are per-request.
- negative_prompt is accepted and ignored in v1.
"""

import base64
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_MIME = "video/mp4"
_SEQ_LOCK = threading.Lock()


def _jobs_dir(server_dir: str) -> Path:
    return Path(server_dir) / "jobs"


def _spool_dir(server_dir: str) -> Path:
    return Path(server_dir) / "spool"


def _server_config(server_dir: str) -> dict:
    p = Path(server_dir) / "server.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_b64_image(server_dir: str, job_id: str, field: str, payload) -> str | None:
    """Decode a base64/data-URL image field to a PNG under uploads/; returns
    the path or None when the field is absent. Raises ValueError on bad data."""
    if not payload:
        return None
    raw = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    import base64 as _b64

    try:
        data = _b64.b64decode(raw, validate=False)
    except Exception as e:
        raise ValueError(f"{field} is not valid base64: {e}")
    if len(data) < 64 or len(data) > 20 * 1024 * 1024:
        raise ValueError(f"{field} size out of range ({len(data)} bytes)")
    updir = Path(server_dir) / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    path = updir / f"{job_id}_{field}.png"
    path.write_bytes(data)
    try:
        from PIL import Image as _Image

        _Image.open(path).convert("RGB").verify()
    except Exception as e:
        path.unlink(missing_ok=True)
        raise ValueError(f"{field} is not a decodable image: {e}")
    return str(path)


class _Handler(BaseHTTPRequestHandler):
    server_dir = None  # set by serve()

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 8 * 1024 * 1024:
            return {}
        return json.loads(self.rfile.read(n))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {"ok": True, "engine": "h3-t2va"})
            return
        if path == "/sdcpp/v1/capabilities":
            cfg = _server_config(self.server_dir)
            self._send_json(
                200,
                {
                    "engine": "h3-t2va",
                    "supported_modes": ["vid_gen"],
                    "output_formats_by_mode": {"vid_gen": ["mp4"]},
                    "defaults_by_mode": {
                        "vid_gen": {
                            "video_frames": cfg.get("video_frames"),
                            "fps": cfg.get("fps", 24),
                            "width": cfg.get("width"),
                            "height": cfg.get("height"),
                            "sample_steps": cfg.get("steps", 4),
                            "note": "video_frames/fps/size are server-wide; prompt/steps/seed per request",
                        }
                    },
                    "features_by_mode": {
                        "vid_gen": {
                            "per_request_prompt": True,
                            "per_request_steps": True,
                            "per_request_seed": True,
                            "per_request_frames": True,
                            "per_request_resolution": "snap to supported buckets",
                            "img2vid": True,
                            "img2vid_note": "init_image = start anchor (fl2va); end_image optional end anchor; portrait requests snap to the canonical landscape bucket",
                            "audio": True,
                        }
                    },
                },
            )
            return
        if path.startswith("/sdcpp/v1/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            jf = _jobs_dir(self.server_dir) / f"{job_id}.json"
            if not jf.exists():
                self._send_json(404, {"error": f"unknown job {job_id}"})
                return
            try:
                job = json.loads(jf.read_text())
            except Exception as e:
                self._send_json(500, {"error": f"job read failed: {e}"})
                return
            if job.get("status") == "completed" and job.get("artifact"):
                ap = Path(job["artifact"])
                if ap.exists():
                    b64 = base64.b64encode(ap.read_bytes()).decode()
                    job["result"] = {
                        "b64_json": b64,
                        "mime_type": _MIME,
                        "output_format": "mp4",
                        "fps": job.get("fps", 24),
                        "frame_count": job.get("frame_count", 0),
                    }
                else:
                    job["status"] = "failed"
                    job["error"] = "artifact missing"
            self._send_json(200, job)
            return
        self._send_json(404, {"error": f"no route {path}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/sdcpp/v1/vid_gen":
            try:
                req = self._read_body_json()
            except Exception as e:
                self._send_json(400, {"error": f"bad json: {e}"})
                return
            prompt = (req.get("prompt") or "").strip()
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return
            fmt = (req.get("output_format") or "mp4").lower()
            if fmt != "mp4":
                self._send_json(400, {"error": f"output_format {fmt!r} unsupported; this backend emits mp4 (H.264+AAC)"})
                return
            spool = _spool_dir(self.server_dir)
            jobs = _jobs_dir(self.server_dir)
            # v101b: monotonic seq from a counter file. The v101 generator
            # counted spool req_*.json — but claimed files are RENAMED to
            # .claimed, so every post-claim request got seq 1 again; rank1's
            # seq-difference check never fired and rank0 ran the request solo
            # (denoise collectives stall -> watchdog kill, observed live).
            with _SEQ_LOCK:
                seq_file = Path(self.server_dir) / ".seq"
                try:
                    last = int(seq_file.read_text() or 0)
                except Exception:
                    last = 0
                seq = last + 1
                _tmp = seq_file.with_suffix(".tmp")
                _tmp.write_text(str(seq))
                _tmp.replace(seq_file)
            job_id = f"job_{secrets.token_hex(5)}"
            record = {
                "id": job_id,
                "seq": seq,
                "kind": "vid_gen",
                "prompt": prompt,
                "negative_prompt": req.get("negative_prompt") or "",
                "steps": int((req.get("sample_params") or {}).get("sample_steps") or _server_config(self.server_dir).get("steps", 4)),
                "seed": int(req.get("seed", -1)),
                "video_frames": req.get("video_frames"),
                "fps": req.get("fps", 24),
                "width": req.get("width"),
                "height": req.get("height"),
                "out_dir": str(Path(self.server_dir).parent),
                "status": "queued",
                "created": int(time.time()),
            }
            # v106: i2va — init_image (start anchor) / end_image (end anchor).
            try:
                record["init_image"] = _save_b64_image(self.server_dir, job_id, "init_image", req.get("init_image"))
                record["end_image"] = _save_b64_image(self.server_dir, job_id, "end_image", req.get("end_image"))
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            tmp = spool / f".req_{seq:06d}_{job_id}.tmp"
            tmp.write_text(json.dumps(record))
            tmp.replace(spool / f"req_{seq:06d}_{job_id}.json")
            (jobs / f"{job_id}.json").write_text(json.dumps(record))
            self._send_json(
                202,
                {
                    "id": job_id,
                    "kind": "vid_gen",
                    "status": "queued",
                    "created": record["created"],
                    "poll_url": f"/sdcpp/v1/jobs/{job_id}",
                },
            )
            return
        if path.startswith("/sdcpp/v1/jobs/") and path.endswith("/cancel"):
            job_id = path.split("/")[-2]
            (Path(self.server_dir) / "spool" / f"{job_id}.cancel").write_text("1")
            self._send_json(202, {"id": job_id, "status": "cancelling"})
            return
        self._send_json(404, {"error": f"no route {path}"})


def serve(server_dir: str, port: int) -> None:
    """Blocking HTTP front-end. Call from the parent main thread (CP>1) or a
    daemon thread (CP=1)."""
    handler = type("BoundHandler", (_Handler,), {"server_dir": server_dir})
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    httpd.serve_forever()


def request_shutdown(server_dir: str) -> None:
    (Path(server_dir) / "SHUTDOWN").write_text("1")
