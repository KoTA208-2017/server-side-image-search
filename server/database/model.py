from sqlalchemy import Column, String, Integer

from . import base
from sqlalchemy_serializer import SerializerMixin

class Product(base.Base, SerializerMixin):
	__tablename__ = 'product'

	id = Column(Integer, primary_key=True)
	name = Column(String)
	price = Column(Integer)
	image = Column(String)
	url = Column(String)
	ecommerce = Column(String)

	def __init__(self, name, price, image, url, ecommerce):
		self.name = name
		self.price = price
		self.image = image
		self.url = url
		self.ecommerce = ecommerce
