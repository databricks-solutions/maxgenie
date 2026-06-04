"""Service layer for Databricks Genie Space operations.

Provides create, update, and get operations via the Databricks SDK.
Uses REST API directly for operations not supported by SDK.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any

from databricks.sdk import WorkspaceClient

from maxgenie.schemas import GenieSpaceConfig

logger = logging.getLogger(__name__)


@dataclass
class GenieSpaceInfo:
    """Genie Space metadata."""
    space_id: str
    title: str
    description: str | None = None
    warehouse_id: str | None = None


class GenieService:
    """Service for interacting with Databricks Genie Spaces."""

    def __init__(self, wc: WorkspaceClient):
        self.wc = wc

    def _api_get(self, path: str, params: dict | None = None) -> dict:
        """Make a GET request to the Databricks API."""
        response = self.wc.api_client.do("GET", path, query=params)
        return response

    def _api_post(self, path: str, body: dict) -> dict:
        """Make a POST request to the Databricks API."""
        response = self.wc.api_client.do("POST", path, body=body)
        return response

    def _api_patch(self, path: str, body: dict) -> dict:
        """Make a PATCH request to the Databricks API."""
        response = self.wc.api_client.do("PATCH", path, body=body)
        return response

    def create(
        self,
        config: GenieSpaceConfig,
        warehouse_id: str | None = None,
        title: str | None = None,
        parent_path: str | None = None,
    ) -> GenieSpaceInfo:
        """Create a new Genie Space from config."""
        wh_id = warehouse_id or config.warehouse_id
        space_title = title or config.title

        if not wh_id:
            raise ValueError("warehouse_id is required for creating a Genie Space")
        if not space_title:
            raise ValueError("title is required for creating a Genie Space")

        body = {
            "warehouse_id": wh_id,
            "title": space_title,
            "serialized_space": json.dumps(config.to_serialized_space()),
        }
        if parent_path or config.parent_path:
            body["parent_path"] = parent_path or config.parent_path

        response = self._api_post("/api/2.0/genie/spaces", body)

        space_info = GenieSpaceInfo(
            space_id=response["space_id"],
            title=response["title"],
            description=response.get("description"),
            warehouse_id=response.get("warehouse_id"),
        )
        logger.info(f"Created Genie Space: {space_info.title} (ID: {space_info.space_id})")
        return space_info

    def update(
        self,
        space_id: str,
        config: GenieSpaceConfig,
        title: str | None = None,
        description: str | None = None,
    ) -> GenieSpaceInfo:
        """Update an existing Genie Space."""
        body = {
            "serialized_space": json.dumps(config.to_serialized_space()),
        }
        if title or config.title:
            body["title"] = title or config.title
        if description or config.description:
            body["description"] = description or config.description

        response = self._api_patch(f"/api/2.0/genie/spaces/{space_id}", body)

        space_info = GenieSpaceInfo(
            space_id=response["space_id"],
            title=response["title"],
            description=response.get("description"),
            warehouse_id=response.get("warehouse_id"),
        )
        logger.info(f"Updated Genie Space: {space_info.title} (ID: {space_info.space_id})")
        return space_info

    def get(self, space_id: str) -> tuple[GenieSpaceInfo, GenieSpaceConfig]:
        """Get a Genie Space and its config."""
        response = self._api_get(
            f"/api/2.0/genie/spaces/{space_id}",
            params={"include_serialized_space": "true"},
        )

        space_info = GenieSpaceInfo(
            space_id=response["space_id"],
            title=response["title"],
            description=response.get("description"),
            warehouse_id=response.get("warehouse_id"),
        )

        serialized_data = json.loads(response.get("serialized_space", "{}"))
        config = GenieSpaceConfig.from_serialized_space(
            serialized_data,
            metadata={
                "space_id": space_info.space_id,
                "title": space_info.title,
                "description": space_info.description,
            },
        )
        return space_info, config

    def delete(self, space_id: str) -> None:
        """Delete a Genie Space."""
        self.wc.api_client.do("DELETE", f"/api/2.0/genie/spaces/{space_id}")
        logger.info(f"Deleted Genie Space: {space_id}")


def get_genie_service(profile: str | None = None) -> GenieService:
    """Factory function to create a GenieService with authenticated client."""
    try:
        wc = WorkspaceClient(profile=profile)
        return GenieService(wc)
    except Exception as e:
        logger.error(f"Failed to connect to Databricks: {e}")
        raise
