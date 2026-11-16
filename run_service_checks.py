import json
import statistics
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000"
LOG_PATH = Path("logs/run.jsonl")
REPORT_DIR = Path("logs")

TEXTS = [
    "The noodles were delicious and comforting.",
    "My parents took me to this diner every Sunday.",
    "Try the sandwiches at the corner café.",
    "The cake looked lovely but tasted dry.",
    "We have cooked this family recipe for generations.",
    "Order the house-made lemonade when you visit.",
    "The staff were kind and the dining room was clean.",
    "I learned to bake bread by watching my aunt.",
    "I recommend booking a table on Friday.",
    "The roasted vegetables were flavorful and fresh.",
]


def request(method, path, body=None):
    data = body
    headers = {}
    if isinstance(body, dict):
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def get_json(path):
    status, body = request("GET", path)
    return status, json.loads(body)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3"}:
        raise SystemExit("Usage: python run_service_checks.py 1|2|3")

    run_number = sys.argv[1]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_start = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0

    # Check the service is reachable before testing predictions.
    startup_status, startup_metrics = get_json("/metrics")

    # Ten valid predictions; capture metrics before deliberately sending errors.
    predictions = []
    for text in TEXTS:
        status, body = request("POST", "/predict", {"text": text})
        predictions.append({"status": status, "body": body})

    metrics_status, metrics = get_json("/metrics")

    # Exercise bad inputs, then verify the service still handles a valid request.
    bad_inputs = []
    for name, body in [
        ("malformed_json", b"{"),
        ("empty_text", json.dumps({"text": ""}).encode("utf-8")),
        ("oversized_body", json.dumps({"text": "x" * 32001}).encode("utf-8")),
    ]:
        status, response_body = request("POST", "/predict", body)
        bad_inputs.append({
            "case": name,
            "status": status,
            "body": response_body,
        })

    after_error_status, after_error_body = request(
        "POST", "/predict", {"text": "The food was delicious."}
    )

    # Read only log records appended during this trial.
    new_bytes = LOG_PATH.read_bytes()[log_start:] if LOG_PATH.exists() else b""
    new_lines = [line for line in new_bytes.decode(
        "utf-8").splitlines() if line]
    records = [json.loads(line) for line in new_lines]

    prediction_records = records[:10]
    log_mean = (
        statistics.mean(record["latency_ms"] for record in prediction_records)
        if len(prediction_records) == 10 else None
    )

    labels = metrics.get("predicted_labels", {})
    checks = {
        "prediction_10_of_10": all(item["status"] == 200 for item in predictions),
        "ten_prediction_logs": (
            len(prediction_records) == 10
            and all(record.get("status") == 200 for record in prediction_records)
        ),
        "no_submitted_text_in_logs": not any(
            text in line for text in TEXTS for line in new_lines
        ),
        "metrics_match_ten_predictions": (
            metrics_status == 200
            and metrics.get("requests") == 10
            and metrics.get("errors") == 0
            and sum(labels.values()) == 10
            and log_mean is not None
            and abs(metrics.get("average_latency_ms", -999) - log_mean) <= 0.1
        ),
        "bad_inputs_and_recovery": (
            [item["status"] for item in bad_inputs] == [400, 400, 400]
            and after_error_status == 200
        ),
        "service_and_metrics_endpoint": startup_status == 200,
    }

    report = {
        "run": int(run_number),
        "startup": {"metrics_status": startup_status, "metrics": startup_metrics},
        "predictions": predictions,
        "metrics_after_ten_predictions": {
            "http_status": metrics_status,
            "response": metrics,
            "log_latency_mean_ms": log_mean,
        },
        "bad_inputs": bad_inputs,
        "valid_request_after_errors": {
            "status": after_error_status,
            "body": after_error_body,
        },
        "new_log_records": records,
        "checks": checks,
    }

    report_path = REPORT_DIR / f"check_run_{run_number}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved evidence to {report_path}")


if __name__ == "__main__":
    main()
