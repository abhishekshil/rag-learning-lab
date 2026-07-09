# neo4j_graph.py
# Citation / entity graph — answers "what improved on X?" (similarity cannot).

from __future__ import annotations

from dataclasses import dataclass

from .citation_data import TUTORIAL_CITATIONS, TUTORIAL_CHUNK_LINKS
from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


@dataclass
class GraphHit:
    """One entity found by walking the citation graph."""
    name: str
    relation: str
    neighbor: str
    chunk_ids: list[str]


class Neo4jCitationGraph:
    """Thin wrapper around LangChain's Neo4jGraph for citation traversal."""

    def __init__(
        self,
        *,
        url: str = NEO4J_URI,
        username: str = NEO4J_USERNAME,
        password: str = NEO4J_PASSWORD,
        reset: bool = False,
    ):
        from langchain_neo4j import Neo4jGraph

        self._graph = Neo4jGraph(url=url, username=username, password=password)
        if reset:
            self.clear()

    def clear(self) -> None:
        self._graph.query("MATCH (n) DETACH DELETE n")

    def seed_tutorial_graph(self) -> int:
        """Create paper nodes, citation edges, and chunk_id links."""
        self.clear()
        for paper in TUTORIAL_CHUNK_LINKS:
            chunk_ids = TUTORIAL_CHUNK_LINKS[paper]
            self._graph.query(
                """
                MERGE (p:Paper {name: $name})
                SET p.chunk_ids = $chunk_ids
                """,
                {"name": paper, "chunk_ids": chunk_ids},
            )
        for src, dst, rel in TUTORIAL_CITATIONS:
            self._graph.query(
                f"""
                MATCH (a:Paper {{name: $src}}), (b:Paper {{name: $dst}})
                MERGE (a)-[r:{rel.upper()}]->(b)
                """,
                {"src": src, "dst": dst},
            )
        return len(TUTORIAL_CITATIONS)

    def expand(self, entity_name: str, hops: int = 1) -> list[GraphHit]:
        """Walk 1-hop citations from a seed paper/entity name."""
        rows = self._graph.query(
            """
            MATCH (seed:Paper)
            WHERE toLower(seed.name) CONTAINS toLower($name)
            MATCH (seed)-[r]->(neighbor:Paper)
            RETURN seed.name AS seed, type(r) AS relation, neighbor.name AS neighbor,
                   neighbor.chunk_ids AS chunk_ids
            LIMIT 10
            """,
            {"name": entity_name},
        )
        hits: list[GraphHit] = []
        for row in rows:
            hits.append(
                GraphHit(
                    name=row["seed"],
                    relation=str(row["relation"]).lower(),
                    neighbor=row["neighbor"],
                    chunk_ids=list(row.get("chunk_ids") or []),
                )
            )
        return hits

    def name(self) -> str:
        return "Neo4jCitationGraph"
