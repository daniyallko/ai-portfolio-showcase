import pytest
from sql.sql_agent import validate_safe_sql, SQLSecurityError

def test_validate_safe_sql_allows_select_queries():
    valid_query = "SELECT name, category FROM projects WHERE status = 'completed';"
    assert validate_safe_sql(valid_query) == True

def test_validate_safe_sql_blocks_destructive_commands():
    dangerous_queries = [
        "DROP TABLE projects;",
        "DELETE FROM projects WHERE id = 1;",
        "UPDATE skills_inventory SET proficiency_level = 'Novice';",
        "INSERT INTO projects (name) VALUES ('Hacked');",
        "SELECT * FROM projects; DROP TABLE documents;",
        "TRUNCATE TABLE project_metrics;"
    ]
    for dq in dangerous_queries:
        with pytest.raises(SQLSecurityError):
            validate_safe_sql(dq)
