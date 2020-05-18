import unittest
import skimage.io

from detector import Detector

detector = Detector("../weight/mask_rcnn_fashion.h5")
image_input = [10,10,20,20]

class TestDetector(unittest.TestCase):
	def test_detection(self):
		# Test with image fashion
		image = skimage.io.imread("test.jpg")
		detection_result = detector.detection(image) 

		output_length = len(detection_result['rois'])
		self.assertGreaterEqual(output_length, 1)

	def test_detection_error(self):							
		image = skimage.io.imread("wall.jpg")
		detection_result = detector.detection(image)   

		output_length = len(detection_result['rois'])
		self.assertEqual(output_length, 0)

	def test_wrong_input(self):		
		self.assertRaises(ValueError, detector.detection, "Wrong input")    

	def test_get_width_of_object(self):		
		width = detector.get_width(image_input)
		self.assertEqual(width, 10)

	def test_get_height_of_object(self):		
		height = detector.get_height(image_input)
		self.assertEqual(height, 10)

	def test_get_area_of_object(self):		
		area = detector.get_area(image_input)
		self.assertEqual(area, 100)
