from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# remotemysql
# engine = create_engine('mysql://iXYqDIic4I:FELalLFBt6@remotemysql.com:3306/iXYqDIic4I')

# db4free.net
engine = create_engine('mysql://tubagus:refdinal123@db4free.net:3306/imagesearch')
Session = sessionmaker(bind=engine)

Base = declarative_base()