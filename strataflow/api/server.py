"""
FastAPI-based API Gateway with GraphQL support.

Provides REST and WebSocket APIs for StrataFlow research engine.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from strataflow.core.config import get_config
from strataflow.core.types import JobID, ResearchRequest
from strataflow.orchestrator.engine import StrataFlowEngine

logger = structlog.get_logger()


# ============================================================================
# Request/Response Models
# ============================================================================


class ResearchJobResponse(BaseModel):
    """Response for research job creation."""

    job_id: str
    status_url: str
    websocket_url: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: str
    progress: float
    current_phase: str | None
    error: str | None
    metrics: dict[str, Any] | None


# ============================================================================
# API Application
# ============================================================================


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app
    """
    config = get_config()

    app = FastAPI(
        title="StrataFlow v2.0 API",
        version="2.0.0",
        description="Neuro-Symbolic Research Engine API",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize engine
    engine = StrataFlowEngine()

    # Store active WebSocket connections
    active_connections: dict[str, list[WebSocket]] = {}

    # ========================================================================
    # REST Endpoints
    # ========================================================================

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "service": "StrataFlow v2.0",
            "status": "operational",
            "version": "2.0.0",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/research/initiate")
    async def initiate_research(request: ResearchRequest) -> ResearchJobResponse:
        """
        Initiate a new research job.

        Args:
            request: Research request parameters

        Returns:
            Job information with tracking URLs
        """
        logger.info("api_research_initiated", topic=request.topic)

        # Start research job
        job_id = await engine.start_research(request)

        return ResearchJobResponse(
            job_id=str(job_id),
            status_url=f"/research/{job_id}/status",
            websocket_url=f"/ws/research/{job_id}",
        )

    @app.get("/research/{job_id}/status")
    async def get_research_status(job_id: str) -> JobStatusResponse:
        """
        Get status of a research job.

        Args:
            job_id: Job identifier

        Returns:
            Current job status and metrics
        """
        try:
            job_uuid = JobID(UUID(job_id))
        except ValueError:
            return JobStatusResponse(
                job_id=job_id,
                status="error",
                progress=0.0,
                current_phase=None,
                error="Invalid job ID format",
                metrics=None,
            )

        status = engine.get_job_status(job_uuid)

        return JobStatusResponse(
            job_id=status.get("job_id", job_id),
            status=status.get("status", "unknown"),
            progress=status.get("progress", 0.0),
            current_phase=status.get("current_phase"),
            error=status.get("error"),
            metrics=status.get("metrics"),
        )

    @app.post("/research/execute")
    async def execute_research_sync(request: ResearchRequest) -> dict[str, Any]:
        """
        Execute research synchronously and return results.

        Args:
            request: Research request

        Returns:
            Complete research results
        """
        logger.info("api_research_execute_sync", topic=request.topic)

        result = await engine.execute_research(request)

        return result

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/research/{job_id}")
    async def research_updates(websocket: WebSocket, job_id: str) -> None:
        """
        WebSocket endpoint for real-time research updates.

        Args:
            websocket: WebSocket connection
            job_id: Job to monitor
        """
        await websocket.accept()

        # Add to active connections
        if job_id not in active_connections:
            active_connections[job_id] = []
        active_connections[job_id].append(websocket)

        try:
            job_uuid = JobID(UUID(job_id))

            # Stream updates
            while True:
                status = engine.get_job_status(job_uuid)

                # Send status update
                await websocket.send_json({
                    "type": "status_update",
                    "data": status,
                })

                # Check if job is complete
                if status.get("status") in ["completed", "failed"]:
                    await websocket.send_json({
                        "type": "job_complete",
                        "data": status,
                    })
                    break

                # Wait before next update
                await asyncio.sleep(1)

        except WebSocketDisconnect:
            logger.info("websocket_disconnected", job_id=job_id)
        finally:
            # Remove from active connections
            if job_id in active_connections:
                active_connections[job_id].remove(websocket)

    # ========================================================================
    # Metrics and Monitoring
    # ========================================================================

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        """
        Prometheus-compatible metrics endpoint.

        Returns:
            System metrics
        """
        # In production: return Prometheus-formatted metrics
        return {
            "active_jobs": len(engine.jobs),
            "registered_agents": len(engine.registered_agents),
        }

    return app


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Run the API server."""
    import uvicorn

    config = get_config()

    uvicorn.run(
        "strataflow.api.server:create_app",
        factory=True,
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level=config.observability.log_level.lower(),
    )


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
