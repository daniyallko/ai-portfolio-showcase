SAMPLE_PROJECTS = [
    {
        "name": "Enterprise RAG AI Chatbot",
        "category": "RAG / Generative AI",
        "description": "Multi-functional RAG assistant with hybrid search, strict citations, and Text-to-SQL analytics.",
        "tech_stack": ["Python", "FastAPI", "PostgreSQL", "pgvector", "LangChain", "Gemini API"],
        "github_url": "https://github.com/daniyallko/ai-project",
        "live_url": "https://portfolio-rag.daniyal.dev",
        "status": "completed",
        "metrics": [
            {"metric_name": "Retrieval Latency Reduction", "metric_value": 45.0, "unit": "%", "impact_description": "Achieved through HNSW indexing and RRF hybrid fusion."},
            {"metric_name": "Answer Faithfulness", "metric_value": 96.5, "unit": "%", "impact_description": "Validated against RAGAS benchmarks."}
        ]
    },
    {
        "name": "Distributed Event Pipeline",
        "category": "Backend & Cloud",
        "description": "High-throughput streaming telemetry pipeline processing millions of event records.",
        "tech_stack": ["Python", "Kafka", "PostgreSQL", "Redis", "Docker"],
        "github_url": "https://github.com/daniyallko/event-pipeline",
        "live_url": None,
        "status": "completed",
        "metrics": [
            {"metric_name": "Throughput Capacity", "metric_value": 15000.0, "unit": "events/sec", "impact_description": "Peak load sustained without dropped messages."}
        ]
    },
    {
        "name": "Cloudflare Edge Assistant",
        "category": "Edge Computing",
        "description": "Ultra-low latency serverless API router with Cloudflare Workers and KV cache.",
        "tech_stack": ["TypeScript", "Cloudflare Workers", "KV", "REST"],
        "github_url": "https://github.com/daniyallko/edge-assistant",
        "live_url": "https://edge.daniyal.dev",
        "status": "completed",
        "metrics": [
            {"metric_name": "P99 Edge Latency", "metric_value": 28.0, "unit": "ms", "impact_description": "Global edge response delivery time."}
        ]
    }
]

SAMPLE_SKILLS = [
    {"skill_name": "Python", "category": "Languages", "proficiency_level": "Expert", "years_experience": 4.5},
    {"skill_name": "FastAPI", "category": "Backend", "proficiency_level": "Expert", "years_experience": 3.5},
    {"skill_name": "PostgreSQL & pgvector", "category": "Databases", "proficiency_level": "Advanced", "years_experience": 4.0},
    {"skill_name": "LangChain & RAG", "category": "AI / ML", "proficiency_level": "Expert", "years_experience": 2.5},
    {"skill_name": "Docker & OCI Cloud", "category": "DevOps", "proficiency_level": "Advanced", "years_experience": 3.0}
]
