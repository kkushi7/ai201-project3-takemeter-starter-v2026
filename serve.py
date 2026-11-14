import json
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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

stats = {"requests": 0, "errors": 0,
         "latency_ms_total": 0.0, "labels": Counter()}


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
        pass  # Use structured request logs instead of the default text log.

    def do_GET(self):
        if self.path != "/metrics":
            return reply(self, 404, {"error": "not_found"})
        count = stats["requests"]
        return reply(self, 200, {
            "requests": count,
            "errors": stats["errors"],
            "average_latency_ms": round(
                stats["latency_ms_total"] / count, 2
            ) if count else 0,
            "predicted_labels": dict(stats["labels"]),
        })

    def do_POST(self):
        started = time.perf_counter()
        status, label, error = 200, None, None
        stats["requests"] += 1

        try:
            if self.path != "/predict":
                status, error = 404, "not_found"
                return reply(self, status, {"error": error})

            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > MAX_BODY_BYTES:
                status, error = 400, "invalid_body_size"
                return reply(self, status, {"error": error})

            data = json.loads(self.rfile.read(size))
            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str) or not text.strip():
                status, error = 400, "text_required"
                return reply(self, status, {"error": error})

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                padding=True, max_length=MAX_LENGTH,
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

        except (ValueError, json.JSONDecodeError):
            status, error = 400, "invalid_json"
            return reply(self, status, {"error": error})
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
