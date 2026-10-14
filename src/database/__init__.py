from src.database.connection import Base, engine


def initialize_database():
    from src.database import models
    Base.metadata.create_all(bind=engine)