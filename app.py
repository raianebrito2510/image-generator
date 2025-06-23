from flask import Flask, render_template, request, jsonify
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import matplotlib
import os

model_path = 'untitled0.py'

app = Flask(__name__)

mnist_data_path = os.path.join('data', 'mnist.npz')
with np.load(mnist_data_path, allow_pickle=True) as f:

    X_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def init():
    number = int(request.form['number'])

    if number < 0 or number > 9:
        return jsonify({'error': 'Only numbers between 0 and 9 are acceptable'})

    indice = np.where(y_train == number)[0]
    samples = np.random.choice(indice, 5, replace=False)

    images = []
    matplotlib.use('Agg')
    for i in samples:
        fig, ax = plt.subplots()
        ax.imshow(X_train[i], cmap="gray")
        ax.axis('Off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        images.append(img_b64)
    return render_template('index.html', number=number, images=images)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
