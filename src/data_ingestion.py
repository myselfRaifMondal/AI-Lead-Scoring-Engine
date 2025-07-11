import os
try:
    from kafka import KafkaConsumer
except ImportError:
    # Handle case where kafka-python is not available
    print("Warning: kafka-python not available. Data ingestion will be disabled.")
    KafkaConsumer = None
import json
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, Integer, String, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Database setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class FeatureVector(Base):
    __tablename__ = 'feature_vectors'

    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    features = Column(Vector(300))
    score = Column(Float)

Base.metadata.create_all(engine)

# Kafka consumer setup
def create_kafka_consumer(topic: str, bootstrap_servers: str) -> KafkaConsumer:
    return KafkaConsumer(topic, 
                         bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                         auto_offset_reset='earliest',
                         enable_auto_commit=True)

def process_messages(consumer: KafkaConsumer, session: sessionmaker):
    for message in consumer:
        data = message.value
        # Extract and process data
        lead_id = data['lead_id']
        features = data['features']
        # Example scoring function (replace with model prediction logic)
        score = func.random()

        # Upsert feature vector
        feature_vector = session.query(FeatureVector).filter_by(lead_id=lead_id).first()
        if not feature_vector:
            feature_vector = FeatureVector(lead_id=lead_id, features=features, score=score)
            session.add(feature_vector)
        else:
            feature_vector.features = features
            feature_vector.score = score

        session.commit()

if __name__ == "__main__":
    topic = os.getenv("KAFKA_TOPIC")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    consumer = create_kafka_consumer(topic, bootstrap_servers)
    session = Session()
    try:
        process_messages(consumer, session)
    finally:
        session.close()
