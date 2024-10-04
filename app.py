from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Kaydedilmiş modeli yükle
with open('bagging_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # AJAX ile gelen veriyi al
        input_array = np.array([float(i) for i in data.split(',')]).reshape(1, -1)  # Veriyi işle
        prediction = model.predict(input_array)[0]  # Modelden tahmin sonucu al
        return jsonify({'prediction': prediction})  # JSON formatında sonucu döndür
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Geliştirme modunda direkt Flask'ın dahili sunucusunu çalıştır
    app.run(debug=True)
