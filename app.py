from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Kaydedilmiş modeli yükle
with open('bagging_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Tahmin işlemi için AJAX isteğini kabul eden route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # AJAX ile gelen veriyi al
        data = request.json['data']
        # Veriyi işleyip numpy array'e dönüştür
        input_array = np.array([float(i) for i in data.split(',')]).reshape(1, -1)
        
        # Modelden tahmin sonucu al
        prediction = model.predict(input_array)[0]
        
        # JSON formatında sonucu döndür
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
