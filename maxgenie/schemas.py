"""Pydantic models for Genie Space configuration.

Adapted from genie-toolkit with extensions for DABs integration.
"""
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


@dataclass
class GenieLoadOptions:
    """Options for loading table metadata from Unity Catalog."""
    include_column_configs: bool = True
    include_example_values: bool = True
    include_value_dictionary: bool = True


class GenieSampleQuestion(BaseModel):
    """A sample question for the Genie Space."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: list[str] | None = None


class GenieColumnConfig(BaseModel):
    """Configuration for a table column.

    Version 1 uses: get_example_values, build_value_dictionary
    Version 2 uses: enable_format_assistance, enable_entity_matching
    """
    column_name: str
    description: Optional[list[str]] = None
    synonyms: Optional[list[str]] = None
    exclude: Optional[bool] = None
    # Version 2 fields
    enable_format_assistance: bool = False
    enable_entity_matching: bool = False
    # Version 1 fields (for backwards compatibility on read)
    get_example_values: Optional[bool] = None
    build_value_dictionary: Optional[bool] = None

    def model_post_init(self, __context) -> None:
        """Convert v1 fields to v2 on load."""
        if self.get_example_values is not None:
            self.enable_format_assistance = self.get_example_values
            self.get_example_values = None
        if self.build_value_dictionary is not None:
            self.enable_entity_matching = self.build_value_dictionary
            self.build_value_dictionary = None


class GenieTableConfig(BaseModel):
    """Configuration for a data source table."""
    identifier: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: Optional[list[str | None]] = None
    column_configs: Optional[list[GenieColumnConfig]] = None

    @classmethod
    def from_table_name(cls, table_name: str) -> "GenieTableConfig":
        return cls(identifier=table_name)


class GenieMetricView(BaseModel):
    """Pre-aggregated metric view configuration."""
    identifier: str
    description: Optional[list[str]] = None
    column_configs: Optional[list[GenieColumnConfig]] = None


class GenieDataSources(BaseModel):
    """Data sources configuration for a Genie Space."""
    tables: list[GenieTableConfig] | None = None
    metric_views: list[GenieMetricView] | None = None

    @classmethod
    def from_table_list(cls, table_names: list[str]) -> "GenieDataSources":
        return cls(tables=[GenieTableConfig.from_table_name(name) for name in table_names])


class GenieExampleSQLParameter(BaseModel):
    """Parameter for an example SQL query."""
    name: str
    type_hint: Optional[str] = None
    description: str | list[str] | None = None
    default_value: str | int | float | bool | list[Any] | dict[str, Any] | None = None


class GenieExampleSQL(BaseModel):
    """Example SQL query for the Genie Space."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: list[str]
    sql: list[str]
    parameters: Optional[list[GenieExampleSQLParameter]] = None
    usage_guidance: Optional[list[str]] = None


class GenieInstruction(BaseModel):
    """Text instruction for the Genie Space."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: list[str] | None = None


class GenieTableJoinSpec(BaseModel):
    """Join specification for a table."""
    identifier: str = Field(default_factory=lambda: uuid.uuid4().hex)
    alias: str


class GenieJoinSpecs(BaseModel):
    """Join specifications between tables."""
    id: str | None = Field(default_factory=lambda: uuid.uuid4().hex)
    left: GenieTableJoinSpec
    right: GenieTableJoinSpec
    sql: list[str]
    comment: Optional[list[str]] = None
    instruction: Optional[list[str]] = None


class GenieSQLSnippet(BaseModel):
    """Reusable SQL snippet."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    alias: str | None = None
    sql: list[str]
    display_name: str
    synonyms: list[str] | None = None
    comment: Optional[list[str]] = None
    instruction: list[str] | None = None


class GenieSQLFunction(BaseModel):
    """Registered SQL function (UDF)."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    identifier: str


class GenieSQLSnippets(BaseModel):
    """Collection of SQL snippets organized by type."""
    filters: list[GenieSQLSnippet] | None = None
    expressions: list[GenieSQLSnippet] | None = None
    measures: list[GenieSQLSnippet] | None = None


class GenieInstructions(BaseModel):
    """Instructions configuration for a Genie Space."""
    text_instructions: list[GenieInstruction] | None = None
    example_question_sqls: list[GenieExampleSQL] | None = None
    join_specs: list[GenieJoinSpecs] | None = None
    sql_functions: list[GenieSQLFunction] | None = None
    sql_snippets: GenieSQLSnippets | None = None


class GenieConfig(BaseModel):
    """Sample questions configuration."""
    sample_questions: list[GenieSampleQuestion] | None = None


class GenieBenchmarkAnswer(BaseModel):
    """Answer format for a benchmark question."""
    format: str  # "SQL"
    content: list[str]


class GenieBenchmarkQuestion(BaseModel):
    """A benchmark question with ground-truth SQL answer."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: list[str]
    answer: list[GenieBenchmarkAnswer] | None = None


class GenieBenchmarks(BaseModel):
    """Benchmarks configuration for evaluating space quality."""
    questions: list[GenieBenchmarkQuestion] | None = None


_SORT_BY_ID_KEYS = frozenset({
    "sample_questions", "text_instructions", "example_question_sqls",
    "join_specs", "filters", "expressions", "measures", "questions",
})

_SORT_BY_IDENTIFIER_KEYS = frozenset({
    "tables", "metric_views",
})

_SORT_BY_COLUMN_NAME_KEYS = frozenset({"column_configs"})

_SORT_BY_NAME_KEYS = frozenset({"parameters"})


def _enforce_constraints(data: dict[str, Any]) -> None:
    """Apply Databricks API constraints: sorting, null filtering, limits.

    Mutates ``data`` in place.
    """
    def _sort_recursive(obj: Any) -> Any:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    # Filter nulls
                    value = [item for item in value if item is not None]
                    # Sort by appropriate key
                    if value and isinstance(value[0], dict):
                        if key in _SORT_BY_ID_KEYS:
                            value = sorted(value, key=lambda x: x.get("id", ""))
                        elif key in _SORT_BY_IDENTIFIER_KEYS:
                            value = sorted(value, key=lambda x: x.get("identifier", ""))
                        elif key in _SORT_BY_COLUMN_NAME_KEYS:
                            value = sorted(value, key=lambda x: x.get("column_name", ""))
                        elif key in _SORT_BY_NAME_KEYS:
                            value = sorted(value, key=lambda x: x.get("name", ""))
                        elif key == "sql_functions":
                            value = sorted(value, key=lambda x: (x.get("id", ""), x.get("identifier", "")))
                    obj[key] = value
                _sort_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                _sort_recursive(item)

    _sort_recursive(data)

    # Enforce max 1 text_instruction
    instr = data.get("instructions")
    if instr and isinstance(instr.get("text_instructions"), list):
        if len(instr["text_instructions"]) > 1:
            instr["text_instructions"] = instr["text_instructions"][:1]

    # Remove SQL snippets with empty sql
    if instr and isinstance(instr.get("sql_snippets"), dict):
        snippets = instr["sql_snippets"]
        for snippet_type in ("filters", "expressions", "measures"):
            items = snippets.get(snippet_type)
            if isinstance(items, list):
                snippets[snippet_type] = [
                    s for s in items
                    if s.get("sql") and any(s["sql"])
                ]


class GenieSpaceConfig(BaseModel):
    """Complete Genie Space configuration.

    This is the top-level model that combines all Genie Space settings.
    Designed to be serialized to/from YAML for version control.
    """
    version: int = 1

    # Space metadata (for DABs and display)
    space_id: str | None = None
    title: str | None = None
    description: str | None = None
    warehouse_id: str | None = None
    parent_path: str | None = None

    # Genie configuration
    config: GenieConfig | None = None
    data_sources: GenieDataSources | None = None
    instructions: GenieInstructions | None = None
    benchmarks: GenieBenchmarks | None = None

    def to_yaml(self, file_path: str) -> None:
        """Serialize the config to YAML file."""
        data = self.model_dump(exclude_none=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, file_path: str) -> "GenieSpaceConfig":
        """Load config from YAML file."""
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_serialized_space(self) -> dict[str, Any]:
        """Convert to Databricks API serialized_space format.

        Applies sorting and constraint enforcement required by the API.
        """
        result = {
            "version": self.version,
            "config": self.config.model_dump(exclude_none=True) if self.config else None,
            "data_sources": self.data_sources.model_dump(exclude_none=True) if self.data_sources else None,
            "instructions": self.instructions.model_dump(exclude_none=True) if self.instructions else None,
        }

        if self.benchmarks:
            result["benchmarks"] = self.benchmarks.model_dump(exclude_none=True)

        _enforce_constraints(result)
        return result

    def replace_table_references(self, mapping: dict[str, str]) -> "GenieSpaceConfig":
        """Replace table identifiers throughout the configuration.

        Useful for cross-workspace promotion (e.g., dev -> prod catalog mapping).

        Args:
            mapping: Dict mapping old identifiers to new ones,
                     e.g. {"dev.schema.table": "prod.schema.table"}

        Returns:
            A new GenieSpaceConfig with replaced identifiers.
        """
        data = deepcopy(self.to_serialized_space())

        # Update data_sources.tables
        data_sources = data.get("data_sources") or {}
        for table in data_sources.get("tables", []):
            if table.get("identifier") in mapping:
                table["identifier"] = mapping[table["identifier"]]

        # Update data_sources.metric_views
        for mv in data_sources.get("metric_views", []):
            if mv.get("identifier") in mapping:
                mv["identifier"] = mapping[mv["identifier"]]

        # Update join_specs
        instructions = data.get("instructions") or {}
        for js in instructions.get("join_specs", []):
            if js.get("left", {}).get("identifier") in mapping:
                js["left"]["identifier"] = mapping[js["left"]["identifier"]]
            if js.get("right", {}).get("identifier") in mapping:
                js["right"]["identifier"] = mapping[js["right"]["identifier"]]

        # Update sql_functions
        for sf in instructions.get("sql_functions", []):
            if sf.get("identifier") in mapping:
                sf["identifier"] = mapping[sf["identifier"]]

        result = GenieSpaceConfig.from_serialized_space(data)
        result.space_id = self.space_id
        result.title = self.title
        result.description = self.description
        result.warehouse_id = self.warehouse_id
        result.parent_path = self.parent_path
        return result

    @classmethod
    def from_serialized_space(cls, data: dict[str, Any], metadata: dict[str, Any] | None = None) -> "GenieSpaceConfig":
        """Create from Databricks API serialized_space format."""
        config = cls(
            version=data.get("version", 1),
            config=GenieConfig(**data["config"]) if data.get("config") else None,
            data_sources=GenieDataSources(**data["data_sources"]) if data.get("data_sources") else None,
            instructions=GenieInstructions(**data["instructions"]) if data.get("instructions") else None,
            benchmarks=GenieBenchmarks(**data["benchmarks"]) if data.get("benchmarks") else None,
        )
        if metadata:
            config.space_id = metadata.get("space_id")
            config.title = metadata.get("title")
            config.description = metadata.get("description")
        return config


class GenieSpaceRegistry(BaseModel):
    """Registry of all Genie Spaces in the project.

    Used for DABs integration and multi-space management.
    """
    spaces: dict[str, str]  # name -> config file path

    @classmethod
    def from_yaml(cls, file_path: str) -> "GenieSpaceRegistry":
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, file_path: str) -> None:
        data = self.model_dump()
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
