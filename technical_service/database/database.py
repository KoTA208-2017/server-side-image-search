# 1 - imports
import json
from sqlalchemy import func
from . import base
from .model import Product
from sqlalchemy.ext.declarative import DeclarativeMeta

class DAO:
	def __init__(self):
		self.session = base.Session()

	def getAll(self):
		products = self.session.query(Product).all()		
		return products		

	def getProduct(self, ids):
		q = self.session.query(Product)
		products = q.filter(Product.id.in_(ids)).order_by(func.field(Product.id, *ids))
		return products

	def insert(self, sitename, name, price, url, image, imageUrl):
		product = Product(sitename, name, price, url, image, imageUrl)
		self.session.add(product)

		self.session.commit()
		self.session.close()
	

        

