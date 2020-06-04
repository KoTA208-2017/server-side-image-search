from sqlalchemy import Column, String, Integer

from . import base
from sqlalchemy_serializer import SerializerMixin

class Product(base.Base, SerializerMixin):
	__tablename__ = 'tbm_product'

	id = Column(Integer, primary_key=True)
	siteName = Column(String)
	name = Column(String)
	price = Column(Integer)
	url = Column(String)
	image = Column(String)
	imageUrl = Column(String)

	def __init__(self, sitename, name, price, url, image, image_url ):
		self.siteName = sitename
		self.name = name
		self.price = price
		self.url = url
		self.image = image
		self.imageUrl = image_url
		
		
