import model
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    samsung= model.price()
    print(samsung)
    return f'<h1>{samsung}</h1>'

if __name__ == '__main__':
    app.run()