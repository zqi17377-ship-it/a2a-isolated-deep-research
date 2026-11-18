"""Corpus package exports."""

from multi_agent_research.corpus.documents import (
    ClaimRecord,
    CorpusDocument,
    build_research_corpus,
    get_corpus_stats,
)
from multi_agent_research.corpus.search import ResearchCorpusIndex

__all__ = [
    "ClaimRecord",
    "CorpusDocument",
    "ResearchCorpusIndex",
    "build_research_corpus",
    "get_corpus_stats",
]
