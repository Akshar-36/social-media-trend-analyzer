import os
import sys
import json
import time
import socket
import threading
import urllib.request
import urllib.error
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from serve import create_server
from pipeline import RecommenderPipeline

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

@pytest.fixture(scope="module")
def pipeline():
    return RecommenderPipeline()

@pytest.fixture(scope="module")
def server_url(pipeline):
    """Start the HTTP server on a free port in a background thread."""
    port = find_free_port()
    server = create_server(host='127.0.0.1', port=port, pipeline=pipeline)
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    url = f"http://127.0.0.1:{port}"
    
    # Wait for health check readiness (up to 5 seconds)
    ready = False
    start_time = time.time()
    while time.time() - start_time < 5:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.1)
            
    assert ready, "Server failed to pass health check within 5 seconds"
    
    yield url
    
    server.shutdown()
    server.server_close()

def http_get(url):
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

def http_post(url, data=None, raw_body=None, headers=None):
    if headers is None:
        headers = {}
    if data is not None:
        body = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
        headers['Content-Length'] = str(len(body))
    elif raw_body is not None:
        body = raw_body
        headers['Content-Length'] = str(len(body))
    else:
        body = b""
        
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

def test_health_check(server_url):
    status_code, body = http_get(f"{server_url}/health")
    assert status_code == 200
    assert body["status"] == "ok"
    assert body["model_version"] == "v1.0"

def test_recommend_valid(server_url):
    payload = {
        "brand_name": "Lumiere Cosmetics",
        "business_description": "Cruelty-free, vegan makeup and skincare.",
        "industry": "beauty",
        "target_audience": "young adults interested in clean beauty",
        "timezone": "America/New_York"
    }
    status_code, body = http_post(f"{server_url}/recommend", data=payload)
    assert status_code == 200
    assert "recommended_topic" in body
    assert "trending_topics" in body
    assert "recommended_hashtags" in body
    assert body["data_basis"]["category"] == "beauty"
    assert body["data_basis"]["category_match"] == "matched"
    assert body["predicted_engagement"]["value"] > 0
    
    # Verify zero falling recommendations
    details = {h["hashtag"].replace("#", ""): h["trend_direction"] for h in body["hashtag_details"]}
    assert details.get(body["recommended_topic"]) != "falling"
    for t in body["trending_topics"]:
        assert details.get(t) != "falling"

def test_recommend_invalid_schema(server_url):
    # Missing required 'industry'
    payload = {
        "brand_name": "Broken Brand",
        "business_description": "Some description",
        "target_audience": "anyone"
    }
    status_code, body = http_post(f"{server_url}/recommend", data=payload)
    assert status_code == 400
    assert "error" in body
    assert body["error"]["code"] == "INVALID_INPUT"

def test_recommend_extra_property(server_url):
    # Extra property forbidden by additionalProperties: false
    payload = {
        "brand_name": "Test Brand",
        "business_description": "Description",
        "industry": "beauty",
        "target_audience": "audience",
        "forbidden_field": 123
    }
    status_code, body = http_post(f"{server_url}/recommend", data=payload)
    assert status_code == 400
    assert "error" in body
    assert body["error"]["code"] == "INVALID_INPUT"

def test_recommend_malformed_json(server_url):
    bad_bytes = b"{bad json string:"
    status_code, body = http_post(f"{server_url}/recommend", raw_body=bad_bytes)
    assert status_code == 400
    assert body["error"]["code"] == "MALFORMED_JSON"

def test_recommend_missing_content_length(server_url):
    req = urllib.request.Request(f"{server_url}/recommend", method='POST')
    # Omit Content-Length
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            status = resp.status
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read().decode('utf-8'))
        
    assert status == 400
    assert body["error"]["code"] == "INVALID_INPUT"

def test_route_not_found(server_url):
    status_code, body = http_get(f"{server_url}/unknown_route")
    assert status_code == 404
    assert body["error"]["code"] == "NOT_FOUND"

def test_method_not_allowed(server_url):
    req = urllib.request.Request(f"{server_url}/recommend", method='PUT')
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            status = resp.status
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read().decode('utf-8'))
        
    assert status == 405
    assert body["error"]["code"] == "METHOD_NOT_ALLOWED"
