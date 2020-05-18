import unittest
import skimage.io

from detector import Detector

detector = Detector("../weight/mask_rcnn_fashion.h5")

class TestDetector(unittest.TestCase):
	def test_detection(self):
		# Test with image fashion
		image = skimage.io.imread("test.jpg")
		r = detector.detection(image) 

		output_length = len(r['rois'])
		self.assertGreaterEqual(output_length, 1)

	def test_detection_error(self):							
		image = skimage.io.imread("wall.jpg")
		r = detector.detection(image)   

		output_length = len(r['rois'])
		self.assertEqual(output_length, 0)