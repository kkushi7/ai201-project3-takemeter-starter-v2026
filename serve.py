import json
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model")
LOG_PATH = Path("logs/run.jsonl")
MAX_BODY_BYTES = 32_000
MAX_LENGTH = 128

if not MODEL_DIR.is_dir():
    raise SystemExit("Missing ./model. Run the notebook training cell first.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

stats = {
    "requests": 0,
    "errors": 0,
    "latency_ms_total": 0.0,
    "labels": Counter(),
}


def write_log(event):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(json.dumps(event) + "\n")


def reply(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"

        if path == "/":
            return reply(self, 200, {
                "service": "TakeMeter",
                "prediction": "POST /predict",
                "metrics": "GET /metrics",
            })

        if path == "/metrics":
            count = stats["requests"]
            return reply(self, 200, {
                "requests": count,
                "errors": stats["errors"],
                "average_latency_ms": round(
                    stats["latency_ms_total"] / count, 2
                ) if count else 0,
                "predicted_labels": dict(stats["labels"]),
            })

        if path == "/predict":
            return reply(self, 405, {"error": "Send POST to /predict."})

        return reply(self, 404, {"error": "not_found"})

    def do_POST(self):
        started = time.perf_counter()
        status, label, error = 200, None, None
        stats["requests"] += 1

        try:
            path = urlsplit(self.path).path.rstrip("/")
            if path != "/predict":
                status, error = 404, "not_found"
                return reply(self, status, {"error": error})

            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0

            if size < 1 or size > MAX_BODY_BYTES:
                status, error = 400, "invalid_body_size"
                return reply(self, status, {"error": error})

            try:
                data = json.loads(self.rfile.read(size))
            except (json.JSONDecodeError, UnicodeDecodeError):
                status, error = 400, "invalid_json"
                return reply(self, status, {"error": error})

            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str) or not text.strip():
                status, error = 400, "text_required"
                return reply(self, status, {"error": error})

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            with torch.inference_mode():
                probabilities = model(**inputs).logits.softmax(dim=-1)[0]

            index = int(probabilities.argmax())
            label = model.config.id2label.get(
                index, model.config.id2label.get(str(index), str(index))
            )
            return reply(self, 200, {
                "label": label,
                "confidence": round(float(probabilities[index]), 6),
            })

        except Exception:
            status, error = 500, "prediction_failed"
            return reply(self, status, {"error": error})

        finally:
            elapsed = (time.perf_counter() - started) * 1000
            stats["latency_ms_total"] += elapsed
            if status >= 400:
                stats["errors"] += 1
            if label:
                stats["labels"][label] += 1
            write_log({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": status,
                "latency_ms": round(elapsed, 2),
                "label": label,
                "error": error,
            })


if __name__ == "__main__":
    print("TakeMeter listening at http://127.0.0.1:8000")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
