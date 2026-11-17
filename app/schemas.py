from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_NODES = 4
MAX_CONTAINERS = 8
MAX_CPU = 100_000

BACKUP_FORMAT = "node-drain-backup"
BACKUP_VERSION = 1

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _clean_label(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field}不能为空")
    if len(value) > 64:
        raise ValueError(f"{field}不能超过 64 个字符")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{field}不能包含控制字符")
    return value


def _clean_id(value: str, field: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} id 只能是 1-64 位字母、数字或 . _ : -")
    return value


def _check_timestamp(value: str, field: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field}必须是 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field}必须带时区")
    return value


class NodeIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    name: str | None = Field(default=None, max_length=64)
    cpu_capacity: int = Field(gt=0, le=MAX_CPU)

    @field_validator("id")
    @classmethod
    def check_id(cls, value: str) -> str:
        if not _ID_RE.fullmatch(value):
            raise ValueError("节点 id 只能是 1-64 位字母、数字或 . _ : -")
        return value

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_label(value, "节点名称")


class ContainerIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    name: str | None = Field(default=None, max_length=64)
    service: str
    min_ready: int = Field(ge=0, le=MAX_CONTAINERS)
    node_id: str
    cpu_request: int = Field(gt=0, le=MAX_CPU)

    @field_validator("id")
    @classmethod
    def check_id(cls, value: str) -> str:
        if not _ID_RE.fullmatch(value):
            raise ValueError("容器 id 只能是 1-64 位字母、数字或 . _ : -")
        return value

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_label(value, "容器名称")

    @field_validator("service")
    @classmethod
    def clean_service(cls, value: str) -> str:
        return _clean_label(value, "所属业务")


class TopologyIn(BaseModel):
    """Payload accepted by the dashboard's JSON import and the REST API."""

    model_config = ConfigDict(extra="forbid")

    nodes: Annotated[list[NodeIn], Field(min_length=1, max_length=MAX_NODES)]
    containers: Annotated[list[ContainerIn], Field(max_length=MAX_CONTAINERS)]

    @model_validator(mode="after")
    def check_topology(self) -> "TopologyIn":
        node_ids = [node.id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("节点 id 重复")
        container_ids = [container.id for container in self.containers]
        if len(set(container_ids)) != len(container_ids):
            raise ValueError("容器 id 重复")

        known_nodes = set(node_ids)
        per_node: dict[str, int] = {node_id: 0 for node_id in node_ids}
        service_replicas: dict[str, int] = {}
        service_min_ready: dict[str, int] = {}
        for container in self.containers:
            if container.node_id not in known_nodes:
                raise ValueError(
                    f"容器 {container.id!r} 引用了不存在的节点 {container.node_id!r}"
                )
            per_node[container.node_id] += container.cpu_request
            service_replicas[container.service] = (
                service_replicas.get(container.service, 0) + 1
            )
            previous = service_min_ready.get(container.service)
            if previous is not None and previous != container.min_ready:
                raise ValueError(
                    f"业务 {container.service!r} 的 min_ready 不一致："
                    f"{previous} 与 {container.min_ready}"
                )
            service_min_ready[container.service] = container.min_ready

        for node in self.nodes:
            if per_node[node.id] > node.cpu_capacity:
                raise ValueError(
                    f"节点 {node.id!r} 超出容量：请求 {per_node[node.id]} > "
                    f"容量 {node.cpu_capacity}"
                )
        for service, minimum in service_min_ready.items():
            replicas = service_replicas[service]
            if minimum > replicas:
                raise ValueError(
                    f"业务 {service!r} 要求 min_ready={minimum}，但只定义了 "
                    f"{replicas} 个容器"
                )
        return self


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    outcome: Literal["ready", "failed"]
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


# --------------------------------------------------------------------------- #
# Backup / restore payload
# --------------------------------------------------------------------------- #


class BackupNode(BaseModel):
    """A node exactly as persisted, including its derived status."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    name: str
    cpu_capacity: int = Field(gt=0, le=MAX_CPU)
    status: Literal["ready", "draining", "drained"]
    position: int = Field(ge=0, le=MAX_NODES)

    @field_validator("id")
    @classmethod
    def check_id(cls, value: str) -> str:
        return _clean_id(value, "节点")

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        return _clean_label(value, "节点名称")


class BackupService(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    min_ready: int = Field(ge=0, le=MAX_CONTAINERS)
    position: int = Field(ge=0, le=MAX_CONTAINERS)

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        return _clean_label(value, "业务名称")


class BackupContainer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    name: str
    service: str
    cpu_request: int = Field(gt=0, le=MAX_CPU)
    node_id: str
    status: Literal["serving"]
    position: int = Field(ge=0, le=MAX_CONTAINERS)

    @field_validator("id")
    @classmethod
    def check_id(cls, value: str) -> str:
        return _clean_id(value, "容器")

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        return _clean_label(value, "容器名称")


class BackupDrain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    node_id: str
    status: Literal["active", "incomplete", "completed"]
    created_at: str
    finished_at: str | None = None

    @field_validator("created_at")
    @classmethod
    def check_created_at(cls, value: str) -> str:
        return _check_timestamp(value, "排空开始时间")

    @field_validator("finished_at")
    @classmethod
    def check_finished_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _check_timestamp(value, "排空结束时间")


class BackupMigration(BaseModel):
    """One attempted replica move, including requests still holding CPU."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    drain_id: int = Field(ge=1)
    container_id: str
    source_node_id: str
    target_node_id: str | None = None
    state: Literal["starting", "completed", "failed", "blocked"]
    reserved_cpu: int = Field(gt=0, le=MAX_CPU)
    error_code: str | None = Field(default=None, max_length=64)
    error: str | None = Field(default=None, max_length=1000)
    created_at: str
    updated_at: str

    @field_validator("created_at")
    @classmethod
    def check_created_at(cls, value: str) -> str:
        return _check_timestamp(value, "迁移创建时间")

    @field_validator("updated_at")
    @classmethod
    def check_updated_at(cls, value: str) -> str:
        return _check_timestamp(value, "迁移更新时间")


class BackupState(BaseModel):
    """The persisted simulation state, without any presentation-only fields."""

    model_config = ConfigDict(extra="forbid")

    nodes: Annotated[list[BackupNode], Field(max_length=MAX_NODES)]
    services: Annotated[list[BackupService], Field(max_length=MAX_CONTAINERS)]
    containers: Annotated[list[BackupContainer], Field(max_length=MAX_CONTAINERS)]
    drains: Annotated[list[BackupDrain], Field(max_length=1000)]
    migrations: Annotated[list[BackupMigration], Field(max_length=10_000)]

    @model_validator(mode="after")
    def check_references(self) -> "BackupState":
        node_ids = [node.id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("节点 id 重复")
        service_names = [service.name for service in self.services]
        if len(set(service_names)) != len(service_names):
            raise ValueError("业务名称重复")
        container_ids = [container.id for container in self.containers]
        if len(set(container_ids)) != len(container_ids):
            raise ValueError("容器 id 重复")
        drain_ids = [drain.id for drain in self.drains]
        if len(set(drain_ids)) != len(drain_ids):
            raise ValueError("排空任务 id 重复")
        migration_ids = [migration.id for migration in self.migrations]
        if len(set(migration_ids)) != len(migration_ids):
            raise ValueError("迁移尝试 id 重复")

        known_nodes = set(node_ids)
        known_services = set(service_names)
        known_containers = set(container_ids)
        known_drains = set(drain_ids)

        for container in self.containers:
            if container.node_id not in known_nodes:
                raise ValueError(
                    f"容器 {container.id!r} 引用了不存在的节点 {container.node_id!r}"
                )
            if container.service not in known_services:
                raise ValueError(
                    f"容器 {container.id!r} 引用了不存在的业务 {container.service!r}"
                )
        for drain in self.drains:
            if drain.node_id not in known_nodes:
                raise ValueError(
                    f"排空 {drain.id} 引用了不存在的节点 {drain.node_id!r}"
                )
        for migration in self.migrations:
            if migration.drain_id not in known_drains:
                raise ValueError(
                    f"迁移 {migration.id} 引用了不存在的排空 {migration.drain_id}"
                )
            if migration.container_id not in known_containers:
                raise ValueError(
                    f"迁移 {migration.id} 引用了不存在的容器 "
                    f"{migration.container_id!r}"
                )
            if migration.source_node_id not in known_nodes:
                raise ValueError(
                    f"迁移 {migration.id} 引用了不存在的源节点 "
                    f"{migration.source_node_id!r}"
                )
            if (
                migration.target_node_id is not None
                and migration.target_node_id not in known_nodes
            ):
                raise ValueError(
                    f"迁移 {migration.id} 引用了不存在的目标节点 "
                    f"{migration.target_node_id!r}"
                )
        return self


class BackupEnvelope(BaseModel):
    """File wrapper: the exported state plus the revision it was taken at."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["node-drain-backup"]
    version: Literal[1]
    exported_at: str
    revision: int = Field(ge=0)
    state: BackupState

    @field_validator("exported_at")
    @classmethod
    def check_exported_at(cls, value: str) -> str:
        return _check_timestamp(value, "导出时间")
