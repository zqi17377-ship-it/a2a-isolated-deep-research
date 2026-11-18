"""Hybrid TF-IDF & Tag-Boosted Local Research Repository Search Engine."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from multi_agent_research.a2a.models import CredibilityTier
from multi_agent_research.corpus.documents import CorpusDocument, build_research_corpus


class ResearchCorpusIndex:
    """In-memory multi-source document repository with TF-IDF + lexical retrieval."""

    def __init__(self, documents: Optional[list[CorpusDocument]] = None) -> None:
        self._docs: list[CorpusDocument] = documents if documents is not None else build_research_corpus()
        self._doc_map: dict[str, CorpusDocument] = {d.doc_id: d for d in self._docs}
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]+\b",
        )
        corpus_texts = [
            f"{d.title} {' '.join(d.tags)} {d.domain} {d.source_org} {d.content}"
            for d in self._docs
        ]
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus_texts)

    @property
    def documents(self) -> list[CorpusDocument]:
        return list(self._docs)

    def get_document(self, doc_id: str) -> Optional[CorpusDocument]:
        """Lookup a document by exact `doc_id`."""
        return self._doc_map.get(doc_id)

    def verify_verbatim_quote(self, doc_id: str, quote_span: str) -> bool:
        """Check whether `quote_span` appears verbatim (whitespace-normalized) in `doc_id`."""
        doc = self._doc_map.get(doc_id)
        if doc is None or not quote_span.strip():
            return False
        norm_doc = re.sub(r"\s+", " ", doc.content).strip()
        norm_quote = re.sub(r"\s+", " ", quote_span).strip()
        return norm_quote in norm_doc

    def search(
        self,
        query: str,
        top_k: int = 5,
        prefer_primary_only: bool = False,
        exclude_doc_ids: Optional[set[str]] = None,
    ) -> list[tuple[CorpusDocument, float]]:
        """Retrieve top-k matching documents with hybrid TF-IDF + exact entity/tag boost.

        Note: By default (`prefer_primary_only=False`), retrieval is credibility-agnostic
        (mirroring standard web/RAG search where viral blog/forum posts compete on lexical
        relevance with official spec sheets). When `AdversarialCriticAgent` triggers
        targeted re-retrieval (`prefer_primary_only=True`), unverified rumor sources are
        filtered out to locate authoritative primary/secondary specs.
        """
        q_vec = self._vectorizer.transform([query])
        tfidf_scores = (self._tfidf_matrix @ q_vec.T).toarray().ravel()

        q_tokens = set(re.findall(r"[a-z0-9_\-\.]+", query.lower()))
        scored: list[tuple[CorpusDocument, float]] = []

        for idx, doc in enumerate(self._docs):
            if exclude_doc_ids and doc.doc_id in exclude_doc_ids:
                continue
            if prefer_primary_only and doc.credibility_tier == CredibilityTier.UNVERIFIED_RUMOR:
                continue

            score = float(tfidf_scores[idx])
            # Exact entity/tag lexical overlap boost
            tag_hits = sum(1 for t in doc.tags if t.lower() in q_tokens or any(t.lower() in qt for qt in q_tokens))
            score += 0.14 * tag_hits

            # Metric key entity match boost
            for claim in doc.structured_claims:
                entity_prefix = claim.metric_key.split(".")[0].replace("_", "-")
                if entity_prefix in query.lower() or entity_prefix.replace("-", " ") in query.lower():
                    score += 0.22
                    break

            if prefer_primary_only and doc.credibility_tier == CredibilityTier.PRIMARY_SPEC:
                score += 0.25

            if score > 0.01:
                scored.append((doc, round(score, 4)))

        scored.sort(key=lambda item: (-item[1], item[0].doc_id))
        return scored[:top_k]
