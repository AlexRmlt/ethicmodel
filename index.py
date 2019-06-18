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
    
    with open(r'model/moral_data.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        for row in request.json:
            writer.writerow(list(row.values()))

    # retrain
    sgd.train_model()

    return jsonify('Success!'), 200

@app.route('/train', methods=['GET'])
def train():
    sgd.train_model()
    return jsonify('Success!'), 200

@app.route('/predict', methods=['GET'])
def predict():
    labels = ['No', 'Yes']
    gr, confidence = sgd.predict_class(request.args.get('t'))
    
    try:
        res = {
            "general_rule": labels[gr],
            "confidence": confidence
        }
    except KeyError:
        msg = 'Error predicting general rule'
        logger.warning(msg)
        return jsonify(msg), 500
    
    return jsonify(res), 200

