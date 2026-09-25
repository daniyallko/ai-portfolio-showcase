import uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Integer, Numeric, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from pgvector.sqlalchemy import Vector
from core.config import settings

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    tech_stack: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    github_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    live_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="completed")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    metrics: Mapped[list["ProjectMetric"]] = relationship("ProjectMetric", back_populates="project", cascade="all, delete-orphan")

class ProjectMetric(Base):
    __tablename__ = "project_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    unit: Mapped[str] = mapped_column(String(30), nullable=False)
    impact_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship("Project", back_populates="metrics")

class SkillInventory(Base):
    __tablename__ = "skills_inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    skill_name: Mapped[str] = mapped_column(String(50), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    proficiency_level: Mapped[str] = mapped_column(String(30), nullable=False)
    years_experience: Mapped[float] = mapped_column(Numeric(3, 1), nullable=False)

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSIONS), nullable=True)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
