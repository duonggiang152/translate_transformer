from flask import Flask, request, jsonify, render_template, redirect, url_for
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from positional_encoding_layer import PositionalEncodingLayer
import tensorflow as tf
app = Flask(__name__)

model_translate = tf.keras.models.load_model('saved_model.keras')
vectorizer_model = tf.keras.models.load_model("text_vectorizer.keras")

# Extract the TextVectorization layer
text_vec_layer_vn = vectorizer_model.layers[0]

# Load vocabulary
with open("vectorizer_vocab.txt", "r") as f:
    vocab = [line.strip() for line in f]

# Set the vocabulary in the new layer
text_vec_layer_vn.set_vocabulary(vocab)

def translate_text(sentence_en):
    translation = ""
    for word_idx in range(50):
        X = np.array([sentence_en]) # encoder input
        X_dec = np.array(["startofseq " + translation]) # decoder input
        y_proba = model_translate((X, X_dec))[0, word_idx] # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_vn.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/<path:any_path>')
def catch_all(any_path):
    return redirect(url_for('index'))



@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    translated_text = translate_text(text)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)