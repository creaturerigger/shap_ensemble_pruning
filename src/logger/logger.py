from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime    # type: ignore
from sqlalchemy.ext.declarative import declarative_base    # type: ignore
from sqlalchemy.orm import sessionmaker    # type: ignore
from datetime import datetime
from dotenv import load_dotenv    # type: ignore
import os

load_dotenv()

Base = declarative_base()

class RunLog(Base):
    __tablename__ = "run_logs"
    id = Column(Integer, primary_key=True)
    dataset = Column(String)
    step = Column(String)
    status = Column(String)
    model_path = Column(String, nullable=True)
    feature_path = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)    # type: ignore


class PostgresLogger:
    def __init__(self):
        user_key = "POSTGRES_USER"
        pass_key = "PSQL_PASSWORD"
        db_key = "DB_NAME"
        user = os.getenv(user_key)
        password = os.getenv(pass_key)
        db = os.getenv(db_key)
        db_url = f"postgresql://{user}:{password}@localhost:5432/{db}"
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def log_run(self, dataset, step, status, model_path=None, feature_path=None, accuracy=None):
        with self.Session() as session:
            log = RunLog(
                dataset=dataset,
                step=step,
                status=status,
                model_path=model_path,
                feature_path=feature_path,
                accuracy=accuracy
            )

            session.add(log)
            session.commit()


    def check_if_completed(self, dataset, step):
        with self.Session() as session:
            return session.query(RunLog).filter_by(dataset=dataset, step=step, status="completed").first() is not None
