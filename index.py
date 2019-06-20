from flask import Flask, request, jsonify
import csv
import logging

import gr_sgd as sgd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/append', methods=['POST'])
def append():
    if not request.is_json:
        return 'Invalid input', 400
    
    with open('model/moral-data.csv', 'a') as f:
        writer = csv.DictWriter(f, request.json[0].keys(), delimiter=';')
        writer.writerows(request.json)

    # retrain
    try: 
        sgd.train_model()
        return jsonify('Success!'), 200
    except FileNotFoundError:
        return jsonify('Error training model: no moral data available!'), 500

@app.route('/train', methods=['GET'])
def train():
    try: 
        sgd.train_model()
        return jsonify('Success!'), 200
    except FileNotFoundError:
        return jsonify('Error training model: no moral data available!'), 500

@app.route('/predict', methods=['GET'])
def predict():
    labels = ['No', 'Yes']
    gr, confidence = sgd.predict_class(request.args.get('t'))
    
    try:
        res = {
            "general_rule": labels[gr],
            "confidence": round(confidence, 2)
        }
    except (KeyError, TypeError):
        msg = 'Error predicting general rule'
        logger.warning(msg)
        return jsonify(msg), 500
    
    return jsonify(res), 200

