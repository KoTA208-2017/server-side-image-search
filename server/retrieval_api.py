import os
import h5py
import sys
import skimage.io
import numpy as np
import time

import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify, send_from_directory
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
from database.database import DAO

sys.path.insert(0, "../retrieval")
# from detector import Detector
# import extractor

api = Api(app)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_extension(filename):
	return filename.rsplit('.', 1)[1].lower()

class Retrieval(Resource):	
	def post(self):
		if 'file' not in request.files:
			resp = jsonify({'message' : 'No file part in the request'})
			resp.status_code = 400
			return resp
		file = request.files['file']		
		extension = get_extension(file.filename)		
		# Name file 		
		milli_sec = int(round(time.time() * 1000))		
		filename = secure_filename(str(milli_sec)+"."+extension)	

		# Save image to server
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

api.add_resource(Retrieval, '/retrieval/image', endpoint='image')

if __name__ == "__main__":
    app.run(host= '0.0.0.0', debug=True)	