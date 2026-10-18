import os
import sys
import json
import logging
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pipeline import RecommenderPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def make_handler(pipeline: RecommenderPipeline):
    class RecommenderHTTPHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, data: dict):
            body = json.dumps(data).encode('utf-8')
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed_path = urlparse(self.path).path
            if parsed_path == '/health':
                self._send_json(200, {
                    "status": "ok",
                    "model_version": "v1.0"
                })
            else:
                self._send_json(404, {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": f"Route '{parsed_path}' not found"
                    }
                })

        def do_POST(self):
            parsed_path = urlparse(self.path).path
            if parsed_path == '/recommend':
                content_length = self.headers.get('Content-Length')
                if not content_length or content_length == '0':
                    self._send_json(400, {
                        "error": {
                            "code": "INVALID_INPUT",
                            "message": "Missing Content-Length header or empty request body"
                        }
                    })
                    return

                try:
                    raw_body = self.rfile.read(int(content_length))
                    payload = json.loads(raw_body.decode('utf-8'))
                except Exception as e:
                    self._send_json(400, {
                        "error": {
                            "code": "MALFORMED_JSON",
                            "message": f"Failed to parse request body as JSON: {str(e)}"
                        }
                    })
                    return

                try:
                    result = pipeline.recommend(payload)
                    if "error" in result:
                        self._send_json(400, result)
                    else:
                        self._send_json(200, result)
                except Exception as e:
                    logging.exception("Unhandled error during recommendation:")
                    self._send_json(500, {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": f"Internal pipeline error: {str(e)}"
                        }
                    })
            else:
                self._send_json(404, {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": f"Route '{parsed_path}' not found"
                    }
                })

        def do_PUT(self):
            self._send_json(405, {"error": {"code": "METHOD_NOT_ALLOWED", "message": "Method not allowed"}})

        def do_DELETE(self):
            self._send_json(405, {"error": {"code": "METHOD_NOT_ALLOWED", "message": "Method not allowed"}})

        def log_message(self, format, *args):
            logging.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

    return RecommenderHTTPHandler

def create_server(host: str = '0.0.0.0', port: int = 8080, pipeline: RecommenderPipeline = None):
    if pipeline is None:
        pipeline = RecommenderPipeline()
    handler_class = make_handler(pipeline)
    server = HTTPServer((host, port), handler_class)
    return server

def run_server(host: str = '0.0.0.0', port: int = 8080):
    logging.info(f"Starting Recommender Service on http://{host}:{port}...")
    pipeline = RecommenderPipeline()
    server = create_server(host, port, pipeline)
    logging.info(f"Recommender Service listening on http://{host}:{port} (Health check: GET http://{host}:{port}/health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
    finally:
        server.server_close()
        logging.info("Server terminated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Social Media Recommendation Pipeline HTTP Daemon")
    default_host = os.environ.get('RECOMMENDER_HOST', os.environ.get('HOST', '0.0.0.0'))
    default_port = int(os.environ.get('PORT', os.environ.get('RECOMMENDER_PORT', 8080)))
    parser.add_argument('--host', default=default_host, help="Host to bind (default: 0.0.0.0)")
    parser.add_argument('--port', type=int, default=default_port, help="Port to listen (default: 8080 or PORT)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
