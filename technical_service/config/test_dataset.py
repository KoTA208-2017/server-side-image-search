import unittest

from dataset import FashionDataset

class TestFashionDataset(unittest.TestCase):
	def test_dataset(self):
		# Test when directory dataset doesn't exist
		dataset = FashionDataset()		
		self.assertRaises(ValueError, dataset.load_data, "test.json", "datasets")