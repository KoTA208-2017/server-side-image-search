import unittest
import skimage.io

from detector import Detector

detector = Detector("../weight/mask_rcnn_fashion.h5")

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