from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = None
Session = None


class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)  # when
    filename = Column(String(255))
    label = Column(String(20), nullable=False)
    probability_ai = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    latency_ms = Column(Float)
    source = Column(String(10))
    client_ip = Column(String(45))


def init_db(url):
    global engine, Session
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)


def save_prediction(filename, r, source, client_ip=None):
    with Session() as s:
        s.add(Prediction(filename=filename, label=r['label'], probability_ai=r['probability_ai'],
                         confidence=r['confidence'], threshold=r['threshold'], model_name=r['model_name'],
                         model_version=str(r['model_version']), latency_ms=r.get('latency_ms'),
                         source=source, client_ip=client_ip))
        s.commit()


def recent_predictions(n=100):
    with Session() as s:
        return s.query(Prediction).order_by(Prediction.created_at.desc()).limit(n).all()


def db_ok():
    try:
        with engine.connect() as c:
            c.execute(text('SELECT 1'))
        return True
    except Exception:
        return False
