import joblib
import numpy as np
import afdetection.utils.paths as path

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array(
        [-1.8475961031504544,
        -2.074363156340376,
        -1.819993295859804,
        -0.7705509354711509,
        -2.007787818422469,
        -1.7384695271345996,
        -1.7677085265313455,
        -1.1920088514227813,
        -1.770877225313074,
        -1.8258946693113314,
        -1.8348627692192918,
        -1.907200035029201,
        1.0442253128827323,
        -5.148086607779322e-15,
        -1.535024574018818]
    )
    
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediction' : list(prediction)})

if __name__=="__main__":
    
    models_DIR = path.models_dir('best_model.pkl')
    model = joblib.load(models_DIR)
    
    app.run(port=8080)