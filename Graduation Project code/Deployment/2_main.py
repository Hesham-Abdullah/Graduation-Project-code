import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template, jsonify
from flask.ext.session import Session
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from data import DataSet

import threading
import cv2
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
# this is a comment
app = Flask(__name__)
sess_flask = Session()

sess = tf.Session()
graph = tf.get_default_graph()
saved_model = os.path.join('..\data\checkpoints\lrcn-images.014-1.641.hdf5')
set_session(sess)
model = load_model(saved_model)
model_frames_num = 30
data = DataSet(seq_length=model_frames_num, class_limit=None)

ALLOWED_EXTENSIONS = set(['txt', 'mp4', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')


@app.route("/predict", methods=["GET","POST"])
def predictions(frames):
	pred_data = {"success": False}
	params = request.json
	
	if params == None:
		params = request.args

	if params != None:
		assert len(frames) == model_frames_num
		with graph.as_default():
			set_session(sess)
			prediction = model.predict(np.expand_dims(frames, axis=0))
			pred_data['prediction'] = data.print_class_from_prediction(np.squeeze(prediction, axis=0))[0][0]
			pred_data['success'] = True
			
	return pred_data

@app.route('/')
def preprocessVideo(path):
	sequence_list = []
	vid_path = os.path.join(path)           
	cap =cv2.VideoCapture(vid_path)
	dim = (300, 300)
	p = ''

	while True:
		ret, frame = cap.read()
		if ret:
			frame = cv2.flip(frame, 1)

			# full_fram = frame.copy()
			full_fram = cv2.resize(frame, (800,600), interpolation = cv2.INTER_AREA)
			frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
			
			sequence_list.append(frame)
			
			if len(sequence_list) == model_frames_num:
				rescaledList = data.rescale_list(sequence_list, model_frames_num)
				p = predictions(rescaledList)['prediction']
				flash(p)
				sequence_list = []
				rescaledList = []

			# cv2.putText(full_fram, p, (250,550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
			# cv2.imshow('sign view', full_fram)
			# key = cv2.waitKey(1)
			key = cv2.waitKey(1)	
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				break
		else:
			break
		
	cap.release()
	cv2.destroyAllWindows()
	return p


@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file-upload-field' not in request.files:
			
			return redirect(request.url)
		file = request.files['file-upload-field']
		if file.filename == '':
			
			return redirect(request.url)
		if file and allowed_file(file.filename):
			
			filename = secure_filename(file.filename)
			path = os.path.join('uploaded', filename) 
			
			file.save(path)
			flash(filename)

			prediction = preprocessVideo(path)
			

			return redirect('/')
		else:

			flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
			return redirect(request.url)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    sess_flask.init_app(app)

    app.debug = True
    app.run()
