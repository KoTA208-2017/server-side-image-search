import unittest
import skimage.io
import numpy as np

from extractor import Extractor

extractor = Extractor()

class TestExtractor(unittest.TestCase):
	def test_input_type(self):
		# Test input type
		self.assertRaises(ValueError, extractor.extract_feat, "Wrong input") 

	def test_input_shape(self):
		image = skimage.io.imread("test.jpg")		
		self.assertRaises(ValueError, extractor.extract_feat, image) 
		
	def test_output_shape(self):	
		image = skimage.io.imread("test.jpg")	
		resized = skimage.transform.resize(image, (224, 224), preserve_range=True)
		output = extractor.extract_feat(resized)

		self.assertEqual(output.shape[0], 4096)