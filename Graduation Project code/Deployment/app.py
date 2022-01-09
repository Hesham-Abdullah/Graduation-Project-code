from flask import Flask

UPLOAD_FOLDER = r"D:\\College\\2020\\GP\\Coding\\python_flask_file_upload\\uploaded"

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
