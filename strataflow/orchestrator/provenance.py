"""
Provenance Tracking System with Merkle DAG and blockchain-inspired audit trails.

Provides cryptographically verifiable lineage tracking for all facts and inferences.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from strataflow.core.types import PropositionID, SourceID

logger = structlog.get_logger()


# ============================================================================
# Merkle DAG Structures
# ============================================================================


@dataclass
class MerkleNode:
    """Node in a Merkle DAG."""

    id: str
    content: dict[str, Any]
    parents: list[str]  # Parent node IDs
    hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: str | None = None  # Cryptographic signature


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a fact or inference."""

    fact_id: PropositionID
    fact_type: str  # 'cited', 'inferred', 'synthesized'
    sources: list[SourceID]
    inference_method: str | None
    dependencies: list[PropositionID]
    confidence: float
    merkle_hash: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Cryptographic Hash Functions
# ============================================================================


class CryptographicHasher:
    """Provides cryptographic hashing for provenance."""

    @staticmethod
    def hash_content(content: dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of content.

        Args:
            content: Dictionary to hash

        Returns:
            Hex-encoded hash string
        """
        # Serialize to canonical JSON
        canonical_json = json.dumps(content, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
        return hash_obj.hexdigest()

    @staticmethod
    def hash_with_parents(content_hash: str, parent_hashes: list[str]) -> str:
        """
        Compute hash incorporating parent hashes (Merkle DAG style).

        Args:
            content_hash: Hash of current content
            parent_hashes: Hashes of parent nodes

        Returns:
            Combined hash
        """
        combined = content_hash + ''.join(sorted(parent_hashes))
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return hash_obj.hexdigest()

    @staticmethod
    def compute_merkle_root(leaf_hashes: list[str]) -> str:
        """
        Compute Merkle tree root from leaf hashes.

        Args:
            leaf_hashes: List of leaf node hashes

        Returns:
            Root hash
        """
        if not leaf_hashes:
            return hashlib.sha256(b'').hexdigest()

        if len(leaf_hashes) == 1:
            return leaf_hashes[0]

        # Build tree level by level
        current_level = leaf_hashes[:]

        while len(current_level) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    # Odd number of nodes - hash with itself
                    combined = current_level[i] + current_level[i]

                hash_obj = hashlib.sha256(combined.encode('utf-8'))
                next_level.append(hash_obj.hexdigest())

            current_level = next_level

        return current_level[0]


# ============================================================================
# Provenance DAG
# ============================================================================


class ProvenanceDAG:
    """
    Directed Acyclic Graph for tracking fact lineage.

    Uses Merkle DAG structure for cryptographic verification.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, MerkleNode] = {}
        self.root_hashes: list[str] = []
        self.hasher = CryptographicHasher()
        self.logger = logger.bind(component="ProvenanceDAG")

    def add_fact(
        self,
        fact_id: str,
        content: dict[str, Any],
        parent_ids: list[str] | None = None,
    ) -> MerkleNode:
        """
        Add a fact to the provenance DAG.

        Args:
            fact_id: Unique fact identifier
            content: Fact content
            parent_ids: IDs of parent facts (dependencies)

        Returns:
            Created Merkle node
        """
        parent_ids = parent_ids or []

        # Compute content hash
        content_hash = self.hasher.hash_content(content)

        # Get parent hashes
        parent_hashes = [
            self.nodes[pid].hash for pid in parent_ids if pid in self.nodes
        ]

        # Compute final hash incorporating parents
        final_hash = self.hasher.hash_with_parents(content_hash, parent_hashes)

        # Create node
        node = MerkleNode(
            id=fact_id,
            content=content,
            parents=parent_ids,
            hash=final_hash,
        )

        self.nodes[fact_id] = node

        # Update root hashes if this is a leaf
        if not parent_ids:
            self.root_hashes.append(final_hash)

        self.logger.info(
            "fact_added_to_dag",
            fact_id=fact_id,
            hash=final_hash[:16],
            n_parents=len(parent_ids),
        )

        return node

    def verify_integrity(self) -> bool:
        """
        Verify cryptographic integrity of the entire DAG.

        Returns:
            True if all hashes are valid
        """
        for node_id, node in self.nodes.items():
            # Recompute hash
            content_hash = self.hasher.hash_content(node.content)
            parent_hashes = [
                self.nodes[pid].hash for pid in node.parents if pid in self.nodes
            ]
            expected_hash = self.hasher.hash_with_parents(content_hash, parent_hashes)

            if expected_hash != node.hash:
                self.logger.error(
                    "integrity_violation",
                    node_id=node_id,
                    expected=expected_hash[:16],
                    actual=node.hash[:16],
                )
                return False

        self.logger.info("integrity_verified", n_nodes=len(self.nodes))
        return True

    def get_lineage(self, fact_id: str) -> list[MerkleNode]:
        """
        Get complete lineage (ancestry) of a fact.

        Args:
            fact_id: Fact to trace

        Returns:
            List of ancestor nodes in topological order
        """
        lineage: list[MerkleNode] = []
        visited: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in visited or node_id not in self.nodes:
                return

            visited.add(node_id)
            node = self.nodes[node_id]

            # Visit parents first (topological order)
            for parent_id in node.parents:
                dfs(parent_id)

            lineage.append(node)

        dfs(fact_id)
        return lineage

    def export_subgraph(self, fact_ids: list[str]) -> dict[str, Any]:
        """
        Export a subgraph containing specified facts and their lineage.

        Args:
            fact_ids: Facts to include

        Returns:
            Serializable subgraph dictionary
        """
        # Get all relevant nodes
        all_nodes: set[str] = set()
        for fact_id in fact_ids:
            lineage = self.get_lineage(fact_id)
            all_nodes.update(node.id for node in lineage)

        # Export nodes
        exported_nodes = {
            node_id: {
                "content": self.nodes[node_id].content,
                "parents": self.nodes[node_id].parents,
                "hash": self.nodes[node_id].hash,
                "timestamp": self.nodes[node_id].timestamp.isoformat(),
            }
            for node_id in all_nodes
        }

        return {
            "nodes": exported_nodes,
            "root_facts": fact_ids,
        }


# ============================================================================
# Blockchain-Inspired Audit Trail
# ============================================================================


@dataclass
class AuditBlock:
    """Block in the audit trail blockchain."""

    index: int
    timestamp: datetime
    records: list[ProvenanceRecord]
    previous_hash: str
    hash: str
    nonce: int = 0  # For proof-of-work (optional)


class AuditTrailBlockchain:
    """
    Blockchain-inspired audit trail for research provenance.

    Provides tamper-evident logging of all research operations.
    """

    def __init__(self, enable_proof_of_work: bool = False) -> None:
        """
        Initialize audit trail.

        Args:
            enable_proof_of_work: Enable proof-of-work for blocks
        """
        self.chain: list[AuditBlock] = []
        self.pending_records: list[ProvenanceRecord] = []
        self.hasher = CryptographicHasher()
        self.enable_proof_of_work = enable_proof_of_work
        self.logger = logger.bind(component="AuditTrailBlockchain")

        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create the genesis (first) block."""
        genesis = AuditBlock(
            index=0,
            timestamp=datetime.utcnow(),
            records=[],
            previous_hash="0" * 64,
            hash="",
        )

        genesis.hash = self._compute_block_hash(genesis)
        self.chain.append(genesis)

    def _compute_block_hash(self, block: AuditBlock) -> str:
        """Compute hash of a block."""
        content = {
            "index": block.index,
            "timestamp": block.timestamp.isoformat(),
            "records": [
                {
                    "fact_id": str(r.fact_id),
                    "fact_type": r.fact_type,
                    "merkle_hash": r.merkle_hash,
                }
                for r in block.records
            ],
            "previous_hash": block.previous_hash,
            "nonce": block.nonce,
        }

        return self.hasher.hash_content(content)

    def add_record(self, record: ProvenanceRecord) -> None:
        """
        Add a provenance record to pending records.

        Args:
            record: Provenance record to add
        """
        self.pending_records.append(record)
        self.logger.debug("record_added_to_pending", fact_id=str(record.fact_id))

    def mine_block(self, max_records: int = 100) -> AuditBlock:
        """
        Mine a new block with pending records.

        Args:
            max_records: Maximum records per block

        Returns:
            Newly mined block
        """
        # Get records for this block
        records = self.pending_records[:max_records]
        self.pending_records = self.pending_records[max_records:]

        # Create block
        block = AuditBlock(
            index=len(self.chain),
            timestamp=datetime.utcnow(),
            records=records,
            previous_hash=self.chain[-1].hash,
            hash="",
        )

        # Proof of work if enabled
        if self.enable_proof_of_work:
            block = self._proof_of_work(block, difficulty=4)
        else:
            block.hash = self._compute_block_hash(block)

        # Add to chain
        self.chain.append(block)

        self.logger.info(
            "block_mined",
            index=block.index,
            n_records=len(records),
            hash=block.hash[:16],
        )

        return block

    def _proof_of_work(self, block: AuditBlock, difficulty: int = 4) -> AuditBlock:
        """
        Perform proof-of-work to mine block.

        Args:
            block: Block to mine
            difficulty: Number of leading zeros required

        Returns:
            Block with valid proof-of-work
        """
        target = "0" * difficulty

        while True:
            block.hash = self._compute_block_hash(block)

            if block.hash.startswith(target):
                break

            block.nonce += 1

        return block

    def verify_chain(self) -> bool:
        """
        Verify integrity of the entire blockchain.

        Returns:
            True if chain is valid
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check hash
            recomputed_hash = self._compute_block_hash(current)
            if current.hash != recomputed_hash:
                self.logger.error("invalid_block_hash", index=i)
                return False

            # Check previous hash link
            if current.previous_hash != previous.hash:
                self.logger.error("broken_chain_link", index=i)
                return False

        self.logger.info("blockchain_verified", n_blocks=len(self.chain))
        return True

    def get_record_lineage(self, fact_id: PropositionID) -> list[ProvenanceRecord]:
        """
        Get provenance lineage for a fact from the blockchain.

        Args:
            fact_id: Fact to trace

        Returns:
            List of provenance records in chronological order
        """
        lineage: list[ProvenanceRecord] = []

        for block in self.chain:
            for record in block.records:
                if record.fact_id == fact_id:
                    lineage.append(record)

        return lineage


# ============================================================================
# Main Provenance Tracker
# ============================================================================


class ProvenanceTracker:
    """
    Unified provenance tracking system.

    Combines Merkle DAG and blockchain for comprehensive provenance.
    """

    def __init__(self, enable_blockchain: bool = True) -> None:
        """
        Initialize provenance tracker.

        Args:
            enable_blockchain: Enable blockchain audit trail
        """
        self.dag = ProvenanceDAG()
        self.enable_blockchain = enable_blockchain

        if enable_blockchain:
            self.blockchain = AuditTrailBlockchain()

        self.logger = logger.bind(component="ProvenanceTracker")

    def track_fact(
        self,
        fact_id: PropositionID,
        fact_type: str,
        sources: list[SourceID],
        dependencies: list[PropositionID] | None = None,
        inference_method: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Track provenance of a fact.

        Args:
            fact_id: Fact identifier
            fact_type: Type of fact ('cited', 'inferred', etc.)
            sources: Source identifiers
            dependencies: Dependent facts
            inference_method: Method used for inference
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            Merkle hash of the fact
        """
        dependencies = dependencies or []
        metadata = metadata or {}

        # Create content for DAG
        content = {
            "fact_id": str(fact_id),
            "fact_type": fact_type,
            "sources": [str(s) for s in sources],
            "inference_method": inference_method,
            "confidence": confidence,
            "metadata": metadata,
        }

        # Add to DAG
        node = self.dag.add_fact(
            fact_id=str(fact_id),
            content=content,
            parent_ids=[str(dep) for dep in dependencies],
        )

        # Create provenance record
        record = ProvenanceRecord(
            fact_id=fact_id,
            fact_type=fact_type,
            sources=sources,
            inference_method=inference_method,
            dependencies=dependencies,
            confidence=confidence,
            merkle_hash=node.hash,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )

        # Add to blockchain if enabled
        if self.enable_blockchain:
            self.blockchain.add_record(record)

            # Mine block periodically
            if len(self.blockchain.pending_records) >= 50:
                self.blockchain.mine_block()

        self.logger.info(
            "fact_tracked",
            fact_id=str(fact_id),
            fact_type=fact_type,
            merkle_hash=node.hash[:16],
        )

        return node.hash

    def verify_fact(self, fact_id: PropositionID) -> bool:
        """
        Verify integrity of a fact and its lineage.

        Args:
            fact_id: Fact to verify

        Returns:
            True if fact is verified
        """
        # Verify DAG integrity
        if not self.dag.verify_integrity():
            return False

        # Verify blockchain if enabled
        if self.enable_blockchain and not self.blockchain.verify_chain():
            return False

        return True

    def get_audit_trail(self, fact_id: PropositionID) -> dict[str, Any]:
        """
        Get complete audit trail for a fact.

        Args:
            fact_id: Fact to audit

        Returns:
            Comprehensive audit information
        """
        # Get DAG lineage
        lineage = self.dag.get_lineage(str(fact_id))

        # Get blockchain records if enabled
        blockchain_records = []
        if self.enable_blockchain:
            blockchain_records = self.blockchain.get_record_lineage(fact_id)

        return {
            "fact_id": str(fact_id),
            "dag_lineage": [
                {
                    "id": node.id,
                    "hash": node.hash,
                    "timestamp": node.timestamp.isoformat(),
                    "parents": node.parents,
                }
                for node in lineage
            ],
            "blockchain_records": [
                {
                    "fact_type": r.fact_type,
                    "sources": [str(s) for s in r.sources],
                    "confidence": r.confidence,
                    "merkle_hash": r.merkle_hash,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in blockchain_records
            ],
            "verified": self.verify_fact(fact_id),
        }


__all__ = [
    "ProvenanceTracker",
    "ProvenanceDAG",
    "AuditTrailBlockchain",
    "MerkleNode",
    "ProvenanceRecord",
    "AuditBlock",
]
