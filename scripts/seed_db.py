import asyncio
from sqlalchemy import text, select
from core.database import engine, async_session_maker
from sql.database_schema import Base, Project, ProjectMetric, SkillInventory
from sql.seed_data import SAMPLE_PROJECTS, SAMPLE_SKILLS
from core.logging import get_logger

logger = get_logger("seed_db")

async def init_and_seed():
    logger.info("Connecting to database...")
    async with engine.begin() as conn:
        logger.info("Enabling pgvector and uuid extensions...")
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        logger.info("Creating database tables...")
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        existing = await session.execute(select(Project))
        if existing.scalars().first() is None:
            logger.info(f"Seeding {len(SAMPLE_PROJECTS)} portfolio projects and metrics...")
            for p_data in SAMPLE_PROJECTS:
                metrics_data = p_data.get("metrics", [])
                project = Project(
                    name=p_data["name"],
                    category=p_data["category"],
                    description=p_data["description"],
                    tech_stack=p_data["tech_stack"],
                    github_url=p_data.get("github_url"),
                    live_url=p_data.get("live_url"),
                    status=p_data.get("status", "completed")
                )
                session.add(project)
                await session.flush()

                for m in metrics_data:
                    metric = ProjectMetric(
                        project_id=project.id,
                        metric_name=m["metric_name"],
                        metric_value=m["metric_value"],
                        unit=m["unit"],
                        impact_description=m.get("impact_description")
                    )
                    session.add(metric)

            logger.info(f"Seeding {len(SAMPLE_SKILLS)} skills...")
            for s in SAMPLE_SKILLS:
                skill = SkillInventory(
                    skill_name=s["skill_name"],
                    category=s["category"],
                    proficiency_level=s["proficiency_level"],
                    years_experience=s["years_experience"]
                )
                session.add(skill)

            await session.commit()
            logger.info("Database seeding successfully completed!")
        else:
            logger.info("Database is already seeded.")

if __name__ == "__main__":
    asyncio.run(init_and_seed())
