"""
feedback/collector.py  — PATCHED FOR WINDOWS (SQLite instead of PostgreSQL)
Uses aiosqlite so you don't need PostgreSQL installed.
"""

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── Use SQLite — no PostgreSQL installation needed ────────────────────────
SQLITE_URL = "sqlite+aiosqlite:///./ragdb.sqlite3"

class Base(DeclarativeBase):
    pass


class FeedbackRecord(Base):
    __tablename__ = "feedback"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)
    correction = Column(Text, nullable=True)
    chunk_ids = Column(Text, nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DPOPair(Base):
    __tablename__ = "dpo_pairs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    chosen = Column(Text, nullable=False)
    rejected = Column(Text, nullable=False)
    source = Column(String, default="user_feedback")
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_async_engine(SQLITE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("SQLite database initialized at ragdb.sqlite3")


class FeedbackCollector:
    async def save_feedback(self, question, answer, rating=None,
                            correction=None, chunk_ids=None, session_id=None):
        import json
        feedback_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as db:
            record = FeedbackRecord(
                id=feedback_id, question=question, answer=answer,
                rating=rating, correction=correction,
                chunk_ids=json.dumps(chunk_ids or []),
                session_id=session_id,
            )
            db.add(record)
            if rating == -1 and correction:
                db.add(DPOPair(question=question, chosen=correction,
                               rejected=answer, source="user_correction"))
            await db.commit()
        return feedback_id

    async def get_stats(self):
        async with AsyncSessionLocal() as db:
            total    = (await db.execute(text("SELECT COUNT(*) FROM feedback"))).scalar() or 0
            positive = (await db.execute(text("SELECT COUNT(*) FROM feedback WHERE rating=1"))).scalar() or 0
            negative = (await db.execute(text("SELECT COUNT(*) FROM feedback WHERE rating=-1"))).scalar() or 0
            dpo      = (await db.execute(text("SELECT COUNT(*) FROM dpo_pairs"))).scalar() or 0
        return {
            "total_feedback": total, "positive": positive,
            "negative": negative, "dpo_pairs": dpo,
            "satisfaction_rate": round(positive / total * 100, 1) if total else 0,
        }

    async def get_dpo_pairs(self, limit=1000):
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                text("SELECT question, chosen, rejected FROM dpo_pairs LIMIT :limit"),
                {"limit": limit}
            )
        return [{"prompt": r[0], "chosen": r[1], "rejected": r[2]} for r in result.fetchall()]
