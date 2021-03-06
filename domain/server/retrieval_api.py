import os
import h5py
import sys
import skimage.io
import numpy as np
import time
import re
from keras.backend import clear_session
import urllib.request
from app import app
from flask_ngrok import run_with_ngrok
from flask import Flask, request, redirect, jsonify, send_from_directory
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

sys.path.insert(0, "../image")
from detector import Detector
import extractor

sys.path.insert(0, "../../technical_service")
from database.database import DAO

run_with_ngrok(app)
api = Api(app)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

image_extractor = extractor.Extractor()
image_detector = Detector("../../weight/mask_rcnn_fashion.h5", "detection")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_extension(filename):
	return filename.rsplit('.', 1)[1].lower()

class Retrieval(Resource):	
	def post(self):
		# no file uploaded
		if 'file' not in request.files:
			response = self.build_response(1, [])
			return response

		file = request.files['file']		
		# not allowed file
		if not (file and allowed_file(file.filename)):
			response = self.build_response(2, [])
			return response

		extension = get_extension(file.filename)		
		# file name 		
		milli_sec = int(round(time.time() * 1000))		
		filename = secure_filename(str(milli_sec)+"."+extension)	

		# make log folder 
		path = os.path.join("../../logs/request", str(milli_sec)) 
		os.mkdir(path)

		# Save image to server
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		data = []	

		# uploaded image
		image_path = 'image/uploads/'+filename				
		# Add Database
		database = DAO()
		# open image
		image = skimage.io.imread(image_path)

		# Object Detection		
		# clear_session()	
		# Check image channel
		if(image.shape[2] == 4):
			image = image[...,:3]

		detection_results = image_detector.detection(image)		

		output_length = len(detection_results['rois'])
		# there is no fashion object			
		if output_length == 0 : 
			response = self.build_response(3,data)
			return response

		# save image
		image_detector.save_image(image, detection_results, path)

		# Dominan Object
		big_object = image_detector.get_biggest_box(detection_results['rois'])
		cropped_object = image_detector.crop_object(image, big_object, path)

		# Extract
		# clear_session()
		
		query_image_feature = image_extractor.extract_feat(cropped_object)
		image_extractor.save_extracted_feat_as_image(query_image_feature, path)

		# similarity
		product_ids = self.calculate_similarity(query_image_feature, path)
		length = len(product_ids)		
		# there is no data			
		if length == 0 :  
			response = self.build_response(3,data)
			return response

		result = database.getProduct(product_ids)
		for res in result:
			data.append(res.to_dict())
			
		# Success
		response = self.build_response(0,data)
		return response

	def calculate_similarity(self, query_feature, log_dir):
		# open the product data extraction file
		path = "../../featureCNN_map.h5"
		h5f = h5py.File(path,'r')
		feats = h5f['feats'][:]
		id = h5f['id'][:]
		h5f.close()	
		
		# similarity
		scores = np.dot(query_feature, feats.T)
		id_rank = self.sort_by_score(id, scores, log_dir)
		
		return id_rank

	def sort_by_score(self, id, scores, log_dir):
		rank_ID = np.argsort(scores)[::-1]
		rank_score = scores[rank_ID]
		id_rank = id[rank_ID]
		# score > 0.7
		rank = np.r_[(rank_score>0.7).nonzero()]
		final_score = rank_score[rank]		

		id_rank = id_rank[rank]
		compare_result = re.sub(r' *\n *', '\n', 
			np.array2string(np.c_[final_score, id_rank], precision=2,  suppress_small=True).replace('[', '').replace(']', '').strip())		
		
		f = open(os.path.join(log_dir, "result.txt"), "w+")
		f.writelines(compare_result)
		return id_rank

	def build_response(self, index_message, data):
		message = ["success", "no file part in the request", 
		"Please check the uploaded image type, only for jpg, png and jpeg", "There is no data"]
		code = [200, 400, 400, 404] 		
		
		return {'data': data,
			'message' : message[index_message]}, code[index_message]

class ImageServer(Resource):
	def get(self, filename):
		return send_from_directory(app.static_folder, filename)


api.add_resource(Retrieval, '/retrieval/image', endpoint='retrieval')
api.add_resource(ImageServer, '/image/<string:filename>', endpoint='image')

if __name__ == "__main__":
    app.run() 