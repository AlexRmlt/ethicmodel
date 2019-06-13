from flask import Flask, request, jsonify
import csv

app = Flask(__name__)

@app.route('/append', methods=['POST'])
def append():
    if not request.is_json:
        return 'Invalid input', 400
    
    with open(r'moral_data.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        for row in request.json:
            writer.writerow(list(row.values()))

    return jsonify('Success!'), 200