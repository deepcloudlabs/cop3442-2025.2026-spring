from typing import Literal, Optional, Dict, Any, List

from pydantic import BaseModel, Field, model_validator

NodeType = Literal[
    "prompt_version",
    "test_result",
    "rule",
    "dataset",
    "metric",
    "component"
]

EdgeType = Literal[
    "modifies_rule",
    "produces",
    "depends_on",
    "evaluates",
    "fails",
    "passes"
]


class Node(BaseModel):
    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    type: NodeType
    attributes: Optional[Dict[str, Any]] = None


class Edge(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    type: EdgeType
    attributes: Optional[Dict[str, Any]] = None


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

    @model_validator(mode="after")
    def validate_graph_integrity(self):
        node_ids = [node.id for node in self.nodes]

        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node ids must be unique.")

        node_id_set = set(node_ids)

        for edge in self.edges:
            if edge.source not in node_id_set:
                raise ValueError(f"Edge source '{edge.source}' is not defined as a node.")

            if edge.target not in node_id_set:
                raise ValueError(f"Edge target '{edge.target}' is not defined as a node.")

            if edge.source == edge.target:
                raise ValueError(f"Self-loop is not allowed: {edge.source} -> {edge.target}")

        return self


class GraphAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    graph: Graph
