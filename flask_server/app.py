import argparse
import json
import signal
import sys

import _jsonnet
import pyhocon
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from req_res_oop import ModelingRequests

if __name__ == '__main__':
    def signal_handler(signal, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", default='./config_files/gpt2-medium.jsonnet', type=str, help="specify the config file"
    )
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.from_dict(json.loads(_jsonnet.evaluate_file(args.config_path)))
    print(" >>>> loading model and server files . . . ")
    requests_obj = ModelingRequests(config)
    print(" >>>> done ")
    app = Flask(__name__)
    CORS(app)


    @app.route('/get_projections/layer/<layer>/dim/<dim>')
    def get_projections(layer, dim):
        return jsonify(requests_obj.get_projections(layer, dim))


    @app.route('/get_data', methods=['POST'])
    def get_data():
        try:
            request_data = request.get_json()
            print(request_data["interventions"])
            return jsonify(requests_obj.send_request_get_response(request_data))
        except:
            return jsonify({})


    @app.route('/generate', methods=['POST'])
    def generate():
        try:
            request_data = request.get_json()
            return jsonify(requests_obj.send_request_get_response_for_generation(request_data))
        except:
            return jsonify({})


    @app.route('/generate_intervened', methods=['POST'])
    def generate_intervened():
        try:
            request_data = request.get_json()
            return jsonify(requests_obj.request2response_generate_intervened(request_data))
        except:
            return jsonify({})

    @app.route("/get_autoencoder_files", methods=["POST"])
    def get_autoencoder_files():
        try:
            rv = requests_obj.get_autoencoder_files()
            return jsonify({"autoencoder_files": rv})
        except Exception as e:
            print("WARN: Error in <get_autoencoder_files>")
            print(e)
            return jsonify({"autoencoder_files": []})

    @app.route("/activate_autoencoder", methods=["POST"])
    def activate_autoencoder():
        try:
            request_data = request.get_json()
            rv = requests_obj.activate_autoencoder(int(request_data["index"]))
            return jsonify({"return_code": rv})
        except Exception as e:
            print("WARN: Error in <activate_autoencoder>")
            print(e)
            return jsonify({"return_code": False})

    @app.route("/get_max_autoencoder_neuron_per_token", methods=["POST"])
    def get_max_autoencoder_neuron_per_token():
        try:
            request_data = request.get_json()
            return jsonify(requests_obj.request2response_get_max_act_neurons(request_data))
        except Exception as e:
            print("WARN: Error in <get_max_autoencoder_neuron_per_token>")
            print(e)
            return jsonify({})

    @app.route("/get_neuron_activation_per_token", methods=["POST"])
    def get_neuron_activation_per_token():
        try:
            request_data = request.get_json()
            return jsonify(requests_obj.request2resonse_get_neuron_act(request_data))
        except Exception as e:
            print("WARN: Error in <get_neuron_activation_per_token>")
            print(e)
            return jsonify({})

    app.run(debug=False, port=config.server_port, host="0.0.0.0")
