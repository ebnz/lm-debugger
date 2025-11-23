import argparse
import json
import signal
import sys
import traceback

import _jsonnet
import pyhocon
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from .req_res_oop import ModelingRequests

if __name__ == '__main__':
    def signal_handler(signal, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", default='./config_files/codellama-7b-instruct.jsonnet', type=str, help="specify the config file"
    )
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(args.config_path)))
    print(" >>>> loading model and server files . . . ")
    requests_obj = ModelingRequests(config)
    print(" >>>> done ")
    app = Flask(__name__)
    CORS(app)

    @app.route('/get_projections/name/<name>/layer/<layer>/dim/<dim>')
    def get_projections(name, layer, dim):
        return jsonify(requests_obj.get_projections(name, layer, dim))

    @app.route('/get_data', methods=['POST'])
    def get_data():
        try:
            request_data = request.get_json()
            print(request_data)
            rv = requests_obj.request2response(request_data)
            return jsonify(rv)
        except Exception:
            print(traceback.print_exc())
            return jsonify({})

    @app.route('/generate', methods=['POST'])
    def generate():
        try:
            request_data = request.get_json()
            print(request_data)
            rv = requests_obj.request2response_for_generation(request_data)
            return jsonify(rv)
        except Exception:
            print(traceback.print_exc())
            return jsonify({})

    app.run(debug=False, port=config.server_port, host="0.0.0.0")
