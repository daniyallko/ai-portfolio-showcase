import pytest
from sql.database_schema import Base, Project, ProjectMetric, SkillInventory, Document, DocumentChunk
from sql.seed_data import SAMPLE_PROJECTS, SAMPLE_SKILLS

def test_models_have_expected_columns():
    assert hasattr(Project, "id")
    assert hasattr(Project, "name")
    assert hasattr(Project, "category")
    assert hasattr(Project, "tech_stack")

    assert hasattr(ProjectMetric, "metric_name")
    assert hasattr(ProjectMetric, "metric_value")

    assert hasattr(SkillInventory, "skill_name")
    assert hasattr(SkillInventory, "proficiency_level")

    assert hasattr(Document, "file_hash")
    assert hasattr(DocumentChunk, "embedding")
    assert hasattr(DocumentChunk, "metadata_json")

def test_sample_seed_data_integrity():
    assert len(SAMPLE_PROJECTS) >= 3
    assert len(SAMPLE_SKILLS) >= 5
    for proj in SAMPLE_PROJECTS:
        assert "name" in proj
        assert "tech_stack" in proj
        assert "metrics" in proj
