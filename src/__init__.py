# src package
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    logger.info(
        "LangSmith tracing enabled (project: %s)",
        os.getenv("LANGCHAIN_PROJECT", "loan-underwriter"),
    )
