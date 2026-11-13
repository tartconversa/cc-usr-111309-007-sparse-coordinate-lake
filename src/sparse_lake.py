"""稀疏坐标湖表：多维整数坐标、显式零与缺测的不可变快照。

本模块只做坐标裁决、稀疏分块、窗口索引、按坐标恢复与重分块，不拟合
物理张量，也不运行量子算法。输入记录中的坐标是轴顺序下的完整坐标，
坐标为整数；未出现的坐标、记录值为 NULL、记录值为 0 是三种不同状态。

块编码：
    块键 key[i] = coordinate[i] // chunk_shape[i]
    块内 local[i] = coordinate[i] % chunk_shape[i]
    边缘块有效形状 shape[i] = min(chunk_shape[i], lengths[i] - key[i] * chunk_shape[i])
    offset = 以有效形状 shape 对 local 做行主序展平

一个块只保存非空块：offsets 是已占用 offset 的升序元组，values 与
revisions 逐项对齐。offset 不在 offsets 中表示 absent；offset 存在且
value 为 None 表示 present_null；否则表示 present_value（包括数值 0）。
目录中不会为未采样坐标建块，也不会为了稀疏查询先生成完整网格。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .coordinate_reader import load as load_input

ABSENT = "absent"
PRESENT_NULL = "present_null"
PRESENT_VALUE = "present_value"
STATE_NAMES = (ABSENT, PRESENT_NULL, PRESENT_VALUE)

Number = int | float


class SparseLakeError(ValueError):
    """坐标湖表的输入或操作违反契约。"""


class InvalidInputError(SparseLakeError):
    """字段类型、维度或配置不合法。"""


class InvalidCoordinateError(SparseLakeError):
    """坐标、窗口或块位置不在合法域内。"""


class RevisionConflictError(SparseLakeError):
    """同一坐标在最高 revision 上出现多个竞争记录。"""

    def __init__(self, coordinate: tuple[int, ...], revision: int, count: int):
        self.coordinate = coordinate
        self.revision = revision
        self.count = count
        super().__init__(
            f"坐标 {coordinate} 在 revision {revision} 有 {count} 条平级记录，拒绝裁决"
        )


class AxisOrderError(SparseLakeError):
    """调用方声明的轴顺序与快照固定轴顺序不一致。"""

    def __init__(self, expected: tuple[str, ...], received: tuple[str, ...]):
        self.expected = expected
        self.received = received
        super().__init__(f"轴顺序不一致：快照为 {expected}，收到 {received}")


# ---------------------------------------------------------------------------
# 基础校验
# ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _names(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidInputError(f"{field} 必须是字符串序列")
    result = tuple(value)
    if not result:
        raise InvalidInputError(f"{field} 不能为空")
    if any(not isinstance(item, str) or not item for item in result):
        raise InvalidInputError(f"{field} 每项必须是非空字符串")
    if len(set(result)) != len(result):
        raise InvalidInputError(f"{field} 不能有重复轴名")
    return result


def _dims(value: object, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidInputError(f"{field} 必须是整数序列")
    result = tuple(value)
    if not result:
        raise InvalidInputError(f"{field} 不能为空")
    if any(not _is_int(item) or item <= 0 for item in result):
        raise InvalidInputError(f"{field} 每项必须是正整数")
    return result


def _ints(value: object, rank: int, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise InvalidInputError(f"{field} 必须是长度为 {rank} 的整数序列")
    result = tuple(value)
    if len(result) != rank:
        raise InvalidInputError(f"{field} 长度为 {len(result)}，应为 {rank}")
    if any(not _is_int(item) for item in result):
        raise InvalidInputError(f"{field} 每项必须是整数")
    return result


def _coordinate(value: object, meta: "SnapshotMeta", field: str = "coordinate") -> tuple[int, ...]:
    result = _ints(value, meta.rank, field)
    for extent, item in zip(meta.lengths, result):
        if item < 0 or item >= extent:
            raise InvalidCoordinateError(
                f"{field} 的坐标 {result} 超出 [0,{extent}) 的合法范围"
            )
    return result


def _window(
    lower: object, upper: object, meta: "SnapshotMeta"
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    low = _ints(lower, meta.rank, "window.lower")
    high = _ints(upper, meta.rank, "window.upper")
    for axis, (lo, hi, extent) in enumerate(zip(low, high, meta.lengths)):
        if not (0 <= lo <= hi <= extent):
            raise InvalidCoordinateError(
                f"窗口轴 {axis} 必须满足 0 <= lower <= upper <= {extent}，收到 [{lo},{hi})"
            )
    return low, high


# ---------------------------------------------------------------------------
# 快照元数据
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotMeta:
    """快照的固定维度顺序、各维长度与块形状。"""

    axes: tuple[str, ...]
    lengths: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    offset_order: str = "row_major"

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", _names(self.axes, "axes"))
        object.__setattr__(self, "lengths", _dims(self.lengths, "lengths"))
        object.__setattr__(self, "chunk_shape", _dims(self.chunk_shape, "chunk_shape"))
        if len(self.axes) != len(self.lengths) or len(self.axes) != len(self.chunk_shape):
            raise InvalidInputError("axes、lengths、chunk_shape 维数必须一致")
        if self.offset_order != "row_major":
            raise InvalidInputError("目前只支持 row_major 展平顺序")

    @property
    def rank(self) -> int:
        return len(self.axes)

    def verify_axes(self, axes: Sequence[str] | None) -> None:
        if axes is None:
            return
        received = tuple(axes)
        if received != self.axes:
            raise AxisOrderError(self.axes, received)

    def contains(self, coordinate: Sequence[int]) -> bool:
        return len(coordinate) == self.rank and all(
            0 <= coordinate[i] < self.lengths[i] for i in range(self.rank)
        )

    def chunk_key(self, coordinate: Sequence[int]) -> tuple[int, ...]:
        """逐维整除 chunk_shape，得到块键。"""
        coord = _coordinate(coordinate, self)
        return tuple(coord[i] // self.chunk_shape[i] for i in range(self.rank))

    def chunk_local(self, coordinate: Sequence[int]) -> tuple[int, ...]:
        """逐维取模 chunk_shape，得到块内坐标。"""
        coord = _coordinate(coordinate, self)
        return tuple(coord[i] % self.chunk_shape[i] for i in range(self.rank))

    def effective_shape(self, key: Sequence[int]) -> tuple[int, ...]:
        """块的全局有效边缘形状；最后一块可以小于完整 chunk_shape。"""
        block_key = _ints(key, self.rank, "block key")
        shape = []
        for i, item in enumerate(block_key):
            if item < 0:
                raise InvalidCoordinateError(f"块键每项必须非负，收到 {block_key}")
            extent = min(
                self.chunk_shape[i], self.lengths[i] - item * self.chunk_shape[i]
            )
            if extent <= 0:
                raise InvalidCoordinateError(f"块键 {block_key} 超出快照范围")
            shape.append(extent)
        return tuple(shape)

    def as_dict(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "lengths": list(self.lengths),
            "chunk_shape": list(self.chunk_shape),
            "offset_order": self.offset_order,
        }


# ---------------------------------------------------------------------------
# 展平与块编码
# ---------------------------------------------------------------------------


def row_major_offset(coordinate: Sequence[int], shape: Sequence[int]) -> int:
    """按行主序把局部坐标展平为 offset。"""
    extents = _dims(shape, "shape")
    local = _ints(coordinate, len(extents), "coordinate")
    for axis, (item, extent) in enumerate(zip(local, extents)):
        if item < 0 or item >= extent:
            raise InvalidCoordinateError(
                f"局部坐标轴 {axis} 的 {item} 超出 [0,{extent})"
            )
    offset = 0
    for extent, item in zip(extents, local):
        offset = offset * extent + item
    return offset


def unflatten_offset(offset: int, shape: Sequence[int]) -> tuple[int, ...]:
    """按行主序把 offset 还原为局部坐标。"""
    extents = _dims(shape, "shape")
    if not _is_int(offset) or offset < 0 or offset >= math.prod(extents):
        raise InvalidCoordinateError(f"offset {offset!r} 超出有效块尺寸 {extents}")
    local = [0] * len(extents)
    for axis in range(len(extents) - 1, -1, -1):
        local[axis] = offset % extents[axis]
        offset //= extents[axis]
    return tuple(local)


def chunk_key(meta: SnapshotMeta, coordinate: Sequence[int]) -> tuple[int, ...]:
    return meta.chunk_key(coordinate)


def chunk_local(meta: SnapshotMeta, coordinate: Sequence[int]) -> tuple[int, ...]:
    return meta.chunk_local(coordinate)


def chunk_effective_shape(meta: SnapshotMeta, key: Sequence[int]) -> tuple[int, ...]:
    return meta.effective_shape(key)


# ---------------------------------------------------------------------------
# 逻辑记录与裁决
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """一个坐标的三种逻辑状态之一。"""

    coordinate: tuple[int, ...]
    state: str
    value: Number | None = None
    revision: int | None = None

    def __post_init__(self) -> None:
        if self.state not in STATE_NAMES:
            raise InvalidInputError(f"未知状态 {self.state!r}")
        if self.state == ABSENT:
            if self.value is not None or self.revision is not None:
                raise InvalidInputError("absent 状态不能带值或 revision")
        else:
            if not _is_int(self.revision) or self.revision < 0:
                raise InvalidInputError("present 状态必须带非负整数 revision")
            if self.state == PRESENT_NULL and self.value is not None:
                raise InvalidInputError("present_null 状态的 value 必须为 null")
            if self.state == PRESENT_VALUE:
                if self.value is None or not _valid_value(self.value):
                    raise InvalidInputError("present_value 状态必须带有限数值")

    @property
    def is_present(self) -> bool:
        return self.state != ABSENT

    @property
    def is_null(self) -> bool:
        return self.state == PRESENT_NULL


@dataclass(frozen=True)
class _Record:
    coordinate: tuple[int, ...]
    value: Number | None
    revision: int
    tombstone: bool


_RECORD_FIELDS = ("coordinate", "value", "revision", "tombstone")


def _normalize_record(record: object, meta: SnapshotMeta, where: str) -> _Record:
    if not isinstance(record, Mapping):
        raise InvalidInputError(f"{where} 必须是对象记录")
    missing = [field for field in _RECORD_FIELDS if field not in record]
    if missing:
        raise InvalidInputError(f"{where} 缺少字段 {missing}")

    coordinate = _coordinate(record["coordinate"], meta, f"{where}.coordinate")

    revision = record["revision"]
    if not _is_int(revision) or revision < 0:
        raise InvalidInputError(f"{where}.revision 必须是非负整数")

    tombstone = record["tombstone"]
    if not isinstance(tombstone, bool):
        raise InvalidInputError(f"{where}.tombstone 必须是布尔值")

    value = record["value"]
    if value is not None and not _valid_value(value):
        raise InvalidInputError(f"{where}.value 必须是有限数值或显式 null")
    if tombstone and value is not None:
        raise InvalidInputError(f"{where} 是 tombstone，value 必须为显式 null")

    return _Record(coordinate=coordinate, value=value, revision=revision, tombstone=tombstone)


def _resolve(
    records: Iterable[object], meta: SnapshotMeta
) -> tuple[dict[tuple[int, ...], Cell], int]:
    """先按坐标和最高 revision 裁决，再交给块构建。"""
    if not isinstance(meta, SnapshotMeta):
        raise InvalidInputError("meta 必须是 SnapshotMeta")
    if isinstance(records, Mapping):
        raise InvalidInputError("records 必须是记录序列，不能是单个对象")
    grouped: dict[tuple[int, ...], list[_Record]] = defaultdict(list)
    count = 0
    for position, item in enumerate(records):
        normalized = _normalize_record(item, meta, f"records[{position}]")
        grouped[normalized.coordinate].append(normalized)
        count += 1

    cells: dict[tuple[int, ...], Cell] = {}
    for coordinate, candidates in grouped.items():
        winning_revision = max(candidate.revision for candidate in candidates)
        top = [c for c in candidates if c.revision == winning_revision]
        if len(top) != 1:
            raise RevisionConflictError(coordinate, winning_revision, len(top))
        winner = top[0]
        if winner.tombstone:
            continue
        state = PRESENT_NULL if winner.value is None else PRESENT_VALUE
        cells[coordinate] = Cell(coordinate, state, winner.value, winner.revision)
    return cells, count


def resolve_records(records: Iterable[object], meta: SnapshotMeta) -> tuple[Cell, ...]:
    """裁决重复修订，返回按坐标排序的全部非 tombstone 逻辑单元格。"""
    cells, _ = _resolve(records, meta)
    return tuple(cells[coordinate] for coordinate in sorted(cells))


# ---------------------------------------------------------------------------
# 不可变块、块位置与快照
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """不可变稀疏块；offsets 标记存在的单元格，values=None 表示显式 NULL。"""

    key: tuple[int, ...]
    shape: tuple[int, ...]
    offsets: tuple[int, ...]
    values: tuple[Number | None, ...]
    revisions: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _dims(self.shape, "block shape"))
        object.__setattr__(self, "key", _ints(self.key, len(self.shape), "block key"))
        if len(self.key) != len(self.shape):
            raise InvalidInputError("块 key 与 shape 维数必须一致")
        if any(item < 0 for item in self.key):
            raise InvalidCoordinateError("块键每项必须非负")
        object.__setattr__(self, "offsets", tuple(self.offsets))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "revisions", tuple(self.revisions))
        if not self.offsets:
            raise InvalidInputError("目录只允许非空块")
        if len(self.offsets) != len(self.values) or len(self.offsets) != len(self.revisions):
            raise InvalidInputError("块 offsets/values/revisions 长度必须一致")
        if tuple(sorted(set(self.offsets))) != self.offsets:
            raise InvalidInputError("块 offsets 必须是严格升序的唯一值")
        size = math.prod(self.shape)
        if any(not _is_int(item) or item < 0 or item >= size for item in self.offsets):
            raise InvalidInputError("块 offset 超出有效边缘形状")
        for value in self.values:
            if value is not None and not _valid_value(value):
                raise InvalidInputError(f"块值为不可接受类型：{value!r}")
        for revision in self.revisions:
            if not _is_int(revision) or revision < 0:
                raise InvalidInputError("块 revision 必须是非负整数")

    def position(self, offset: int) -> int:
        index = bisect.bisect_left(self.offsets, offset)
        if index >= len(self.offsets) or self.offsets[index] != offset:
            raise InvalidCoordinateError(f"offset {offset} 不在块 {self.key} 中")
        return index

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "shape": list(self.shape),
            "offsets": list(self.offsets),
            "values": list(self.values),
            "revisions": list(self.revisions),
        }


@dataclass(frozen=True)
class BlockLocation:
    """坐标索引项：命中块的键和块内行主序 offset。"""

    key: tuple[int, ...]
    offset: int


class Snapshot:
    """不可变稀疏快照：块目录、坐标索引、清单与查询。"""

    __slots__ = (
        "_meta",
        "_blocks",
        "_index",
        "_source_record_count",
        "_digest",
        "_logical_digest",
    )

    def __init__(
        self,
        meta: SnapshotMeta,
        blocks: Mapping[tuple[int, ...], Block],
        index: Mapping[tuple[int, ...], BlockLocation],
        source_record_count: int,
    ):
        if not isinstance(meta, SnapshotMeta):
            raise InvalidInputError("meta 必须是 SnapshotMeta")
        if not _is_int(source_record_count) or source_record_count < 0:
            raise InvalidInputError("source_record_count 必须是非负整数")

        frozen_blocks: dict[tuple[int, ...], Block] = {}
        for key, block in blocks.items():
            key_tuple = _ints(key, meta.rank, "block key")
            if not isinstance(block, Block) or block.key != key_tuple:
                raise InvalidInputError(f"块 {key_tuple} 的 key 与目录不一致")
            if block.shape != meta.effective_shape(key_tuple):
                raise InvalidInputError(
                    f"块 {key_tuple} 的 shape {block.shape} 与有效边缘形状不一致"
                )
            frozen_blocks[key_tuple] = block

        frozen_index: dict[tuple[int, ...], BlockLocation] = {}
        for coordinate, location in index.items():
            coord = _coordinate(coordinate, meta, "index coordinate")
            if not isinstance(location, BlockLocation):
                raise InvalidInputError(f"坐标 {coord} 的索引项类型错误")
            block = frozen_blocks.get(location.key)
            if block is None:
                raise InvalidInputError(f"坐标 {coord} 指向不存在的块 {location.key}")
            if location.offset not in block.offsets:
                raise InvalidInputError(
                    f"坐标 {coord} 指向块 {location.key} 中未占用的 offset {location.offset}"
                )
            local = unflatten_offset(location.offset, block.shape)
            block_coordinate = tuple(
                location.key[i] * meta.chunk_shape[i] + local[i]
                for i in range(meta.rank)
            )
            if block_coordinate != coord:
                raise InvalidInputError(
                    f"索引不一致：{coord} 解码为 {block_coordinate}，offset={location.offset}"
                )
            frozen_index[coord] = location

        stored = sum(len(block.offsets) for block in frozen_blocks.values())
        if stored != len(frozen_index):
            raise InvalidInputError("坐标索引项数与块内偏移项数不一致")
        if len(frozen_index) > source_record_count:
            raise InvalidInputError("现存坐标数不能超过源记录数")

        ordered_blocks = {key: frozen_blocks[key] for key in sorted(frozen_blocks)}
        ordered_index = {key: frozen_index[key] for key in sorted(frozen_index)}

        object.__setattr__(self, "_meta", meta)
        object.__setattr__(self, "_blocks", MappingProxyType(ordered_blocks))
        object.__setattr__(self, "_index", MappingProxyType(ordered_index))
        object.__setattr__(self, "_source_record_count", source_record_count)
        object.__setattr__(
            self,
            "_digest",
            _digest_of(meta, ordered_blocks),
        )
        object.__setattr__(
            self,
            "_logical_digest",
            _logical_digest_of(meta, ordered_index, ordered_blocks),
        )
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Snapshot 不可变")

    @classmethod
    def from_cells(
        cls,
        meta: SnapshotMeta,
        cells: Mapping[Sequence[int], Cell] | Iterable[Cell],
        *,
        source_record_count: int | None = None,
    ) -> "Snapshot":
        """从已裁决的逻辑单元格组装快照；不会再合并或求和。"""
        if isinstance(cells, Mapping):
            for key, item in cells.items():
                if not isinstance(item, Cell):
                    raise InvalidInputError("cells 必须包含 Cell")
                if _ints(key, meta.rank, "cells key") != _coordinate(item.coordinate, meta, "cell coordinate"):
                    raise InvalidInputError(f"cells 键 {key} 与单元格坐标 {item.coordinate} 不一致")
            cell_items = list(cells.values())
        else:
            cell_items = list(cells)
        normalized: dict[tuple[int, ...], Cell] = {}
        for item in cell_items:
            if not isinstance(item, Cell):
                raise InvalidInputError("cells 必须包含 Cell")
            coord = _coordinate(item.coordinate, meta, "cell coordinate")
            if coord in normalized:
                raise InvalidInputError(f"cells 中坐标 {coord} 重复")
            if item.state == ABSENT:
                continue
            normalized[coord] = Cell(coord, item.state, item.value, item.revision)
        if source_record_count is None:
            source_record_count = len(normalized)
        return _assemble(meta, normalized, source_record_count)

    @property
    def meta(self) -> SnapshotMeta:
        return self._meta

    @property
    def blocks(self) -> Mapping[tuple[int, ...], Block]:
        return self._blocks

    @property
    def index(self) -> Mapping[tuple[int, ...], BlockLocation]:
        return self._index

    @property
    def source_record_count(self) -> int:
        return self._source_record_count

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def logical_digest(self) -> str:
        return self._logical_digest

    @property
    def cell_count(self) -> int:
        return len(self._index)

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return (
            f"Snapshot(axes={self._meta.axes}, lengths={self._meta.lengths}, "
            f"chunk_shape={self._meta.chunk_shape}, blocks={self.block_count}, "
            f"cells={self.cell_count}, digest={self._digest[:12]})"
        )

    def restore(
        self, coordinate: Sequence[int], *, axes: Sequence[str] | None = None
    ) -> Cell:
        """按完整坐标恢复三种状态之一；越界坐标拒绝而不是返回 absent。"""
        self._meta.verify_axes(axes)
        coord = _coordinate(coordinate, self._meta)
        location = self._index.get(coord)
        if location is None:
            return Cell(coord, ABSENT)
        block = self._blocks[location.key]
        position = block.position(location.offset)
        value = block.values[position]
        state = PRESENT_NULL if value is None else PRESENT_VALUE
        return Cell(coord, state, value, block.revisions[position])

    def __getitem__(self, coordinate: Sequence[int]) -> Cell:
        return self.restore(coordinate)

    def window_plan(
        self,
        lower: Sequence[int],
        upper: Sequence[int],
        *,
        axes: Sequence[str] | None = None,
    ) -> "WindowPlan":
        """只为已存在的非空块规划矩形半开窗口交集，不枚举完整块网格。"""
        self._meta.verify_axes(axes)
        low, high = _window(lower, upper, self._meta)
        plans: list[BlockPlan] = []
        if all(lo < hi for lo, hi in zip(low, high)):
            for key in sorted(self._blocks):
                block = self._blocks[key]
                block_lower = tuple(
                    key[i] * self._meta.chunk_shape[i] for i in range(self._meta.rank)
                )
                block_upper = tuple(
                    block_lower[i] + block.shape[i] for i in range(self._meta.rank)
                )
                intersection_lower = tuple(
                    max(low[i], block_lower[i]) for i in range(self._meta.rank)
                )
                intersection_upper = tuple(
                    min(high[i], block_upper[i]) for i in range(self._meta.rank)
                )
                if all(
                    intersection_lower[i] < intersection_upper[i]
                    for i in range(self._meta.rank)
                ):
                    plans.append(
                        BlockPlan(
                            key=key,
                            effective_shape=block.shape,
                            intersection_lower=intersection_lower,
                            intersection_upper=intersection_upper,
                            stored_offsets=len(block.offsets),
                        )
                    )
        return WindowPlan(lower=low, upper=high, blocks=tuple(plans))

    def scan_window(
        self,
        lower: Sequence[int],
        upper: Sequence[int],
        *,
        axes: Sequence[str] | None = None,
    ) -> tuple[Cell, ...]:
        """扫描窗口内已记录的坐标；absent 保持隐式，不生成密集网格。"""
        plan = self.window_plan(lower, upper, axes=axes)
        rows: list[Cell] = []
        for block_plan in plan.blocks:
            block = self._blocks[block_plan.key]
            for offset in block.offsets:
                local = unflatten_offset(offset, block.shape)
                coordinate = tuple(
                    block_plan.key[i] * self._meta.chunk_shape[i] + local[i]
                    for i in range(self._meta.rank)
                )
                if not all(
                    plan.lower[i] <= coordinate[i] < plan.upper[i]
                    for i in range(self._meta.rank)
                ):
                    continue
                value = block.values[block.position(offset)]
                state = PRESENT_NULL if value is None else PRESENT_VALUE
                rows.append(
                    Cell(coordinate, state, value, block.revisions[block.position(offset)])
                )
        rows.sort(key=lambda cell: cell.coordinate)
        return tuple(rows)

    def rechunk(
        self,
        chunk_shape: Sequence[int],
        *,
        axes: Sequence[str] | None = None,
    ) -> "Snapshot":
        """改变 chunk_shape 后重分块；逻辑坐标到状态的映射保持不变。

        axes 可以显式传入以校验维度顺序；与快照不一致时拒绝而不是猜测。
        """
        if axes is not None:
            self._meta.verify_axes(axes)
        new_meta = SnapshotMeta(
            axes=self._meta.axes,
            lengths=self._meta.lengths,
            chunk_shape=_dims(chunk_shape, "chunk_shape"),
            offset_order=self._meta.offset_order,
        )
        cells = [self.restore(coordinate) for coordinate in self._index]
        return Snapshot.from_cells(
            new_meta, cells, source_record_count=self._source_record_count
        )

    def manifest(self) -> "SnapshotManifest":
        entries = tuple(
            ManifestBlock(
                key=block.key,
                shape=block.shape,
                stored_offsets=len(block.offsets),
                digest=_digest_of_block(block),
            )
            for block in self._blocks.values()
        )
        return SnapshotManifest(
            axes=self._meta.axes,
            lengths=self._meta.lengths,
            chunk_shape=self._meta.chunk_shape,
            offset_order=self._meta.offset_order,
            source_record_count=self._source_record_count,
            cell_count=self.cell_count,
            block_count=self.block_count,
            blocks=entries,
            digest=self._digest,
            logical_digest=self._logical_digest,
        )

    def footprint(self) -> dict[str, Any]:
        """只计算描述性计数，不实例化密集数组。"""
        logical = math.prod(self._meta.lengths)
        recorded = self.cell_count
        nulls = sum(
            1
            for coordinate in self._index
            if self.restore(coordinate).state == PRESENT_NULL
        )
        values = recorded - nulls
        block_grid = math.prod(
            math.ceil(extent / chunk)
            for extent, chunk in zip(self._meta.lengths, self._meta.chunk_shape)
        )
        stored_entries = sum(len(block.offsets) for block in self._blocks.values())
        return {
            "shape": list(self._meta.lengths),
            "chunk_shape": list(self._meta.chunk_shape),
            "logical_cells": logical,
            "recorded_cells": recorded,
            "present_null_cells": nulls,
            "present_value_cells": values,
            "absent_cells": logical - recorded,
            "nonempty_blocks": self.block_count,
            "block_grid_cells": block_grid,
            "stored_entries": stored_entries,
            "dense_export_cells_if_requested": logical,
            "sparse_fraction": (recorded / logical) if logical else 0.0,
            "note": (
                "常见稀疏查询只遍历非空块与已记录坐标；"
                "absent 不占块、不占 offset，也不在扫描中逐格展开。"
            ),
        }

    def dense_export(self) -> "DenseExport":
        """显式请求时才生成密集表示，并附 presence/validity 位图。

        不在任何稀疏查询路径上自动调用；缺失与 NULL 不填 0。
        """
        size = math.prod(self._meta.lengths)
        values: list[Number | None] = [None] * size
        presence = [0] * size
        validity = [0] * size
        for coordinate in self._index:
            cell = self.restore(coordinate)
            flat = row_major_offset(coordinate, self._meta.lengths)
            presence[flat] = 1
            if cell.state == PRESENT_VALUE:
                validity[flat] = 1
                values[flat] = cell.value
        return DenseExport(
            axes=self._meta.axes,
            lengths=self._meta.lengths,
            values=tuple(values),
            presence=tuple(presence),
            validity=tuple(validity),
        )


def _assemble(
    meta: SnapshotMeta,
    cells: Mapping[tuple[int, ...], Cell],
    source_record_count: int,
) -> Snapshot:
    grouped: dict[tuple[int, ...], list[tuple[int, Number | None, int]]] = defaultdict(list)
    index: dict[tuple[int, ...], BlockLocation] = {}
    for coordinate in sorted(cells):
        cell = cells[coordinate]
        if cell.state == ABSENT:
            continue
        key = meta.chunk_key(coordinate)
        local = meta.chunk_local(coordinate)
        shape = meta.effective_shape(key)
        offset = row_major_offset(local, shape)
        grouped[key].append((offset, cell.value, cell.revision))
        location = BlockLocation(key, offset)
        existing = index.get(coordinate)
        if existing is not None and existing != location:
            raise InvalidInputError(f"坐标 {coordinate} 映射到多个块位置")
        index[coordinate] = location

    blocks: dict[tuple[int, ...], Block] = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        offsets = tuple(item[0] for item in entries)
        values = tuple(item[1] for item in entries)
        revisions = tuple(item[2] for item in entries)
        blocks[key] = Block(key=key, shape=meta.effective_shape(key), offsets=offsets, values=values, revisions=revisions)
    return Snapshot(meta, blocks, index, source_record_count)


def build_snapshot(records: Iterable[object], meta: SnapshotMeta) -> Snapshot:
    """从原始逻辑记录裁决并构建不可变稀疏快照。"""
    cells, count = _resolve(records, meta)
    return Snapshot.from_cells(meta, cells, source_record_count=count)


def snapshot_from_config(
    records: Iterable[object],
    config: Mapping[str, Any],
    *,
    expected_axes: Sequence[str] | None = None,
) -> Snapshot:
    """按 tensor 配置构建快照；可显式拒绝轴顺序被调换。"""
    if not isinstance(config, Mapping):
        raise InvalidInputError("config 必须是对象")
    if "axes" not in config or "shape" not in config or "chunk_shape" not in config:
        raise InvalidInputError("config 必须含 axes、shape、chunk_shape")
    axes = _names(config["axes"], "axes")
    if expected_axes is not None:
        expected = _names(expected_axes, "expected_axes")
        if axes != expected:
            raise AxisOrderError(expected, axes)
    meta = SnapshotMeta(
        axes=axes,
        lengths=_dims(config["shape"], "shape"),
        chunk_shape=_dims(config["chunk_shape"], "chunk_shape"),
        offset_order=config.get("offset_order", "row_major"),
    )
    return build_snapshot(records, meta)


def load_snapshot(
    records_path: str | Path,
    config_path: str | Path,
    *,
    expected_axes: Sequence[str] | None = None,
) -> Snapshot:
    """用现有 coordinate_reader 读取原始资料，不覆盖输入文件。"""
    records = load_input(records_path)
    config = load_input(config_path)
    return snapshot_from_config(records, config, expected_axes=expected_axes)


def _digest_of_block(block: Block) -> str:
    payload = json.dumps(block.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_of(meta: SnapshotMeta, blocks: Mapping[tuple[int, ...], Block]) -> str:
    payload = {
        "meta": meta.as_dict(),
        "blocks": [block.to_dict() for _, block in sorted(blocks.items())],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _logical_digest_of(
    meta: SnapshotMeta,
    index: Mapping[tuple[int, ...], BlockLocation],
    blocks: Mapping[tuple[int, ...], Block],
) -> str:
    """逻辑身份摘要：只含轴顺序、长度和坐标→状态，不含分块方式。

    因此改变 chunk_shape 后重分块，逻辑坐标到状态的映射一致时摘要不变。
    """
    cells = []
    for coordinate in sorted(index):
        location = index[coordinate]
        block = blocks[location.key]
        position = block.position(location.offset)
        cells.append(
            {
                "coordinate": list(coordinate),
                "value": block.values[position],
                "revision": block.revisions[position],
            }
        )
    payload = {
        "axes": list(meta.axes),
        "lengths": list(meta.lengths),
        "cells": cells,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 窗口计划与显式密集导出
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockPlan:
    """窗口内某非空块的块内交集与该块有效边缘形状。"""

    key: tuple[int, ...]
    effective_shape: tuple[int, ...]
    intersection_lower: tuple[int, ...]
    intersection_upper: tuple[int, ...]
    stored_offsets: int

    @property
    def intersection_shape(self) -> tuple[int, ...]:
        return tuple(
            high - low
            for low, high in zip(self.intersection_lower, self.intersection_upper)
        )


@dataclass(frozen=True)
class WindowPlan:
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    blocks: tuple[BlockPlan, ...]

    @property
    def block_count(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class DenseExport:
    """刻意请求的密集表示：presence 区分缺测与 NULL，validity 区分 NULL 与数值。"""

    axes: tuple[str, ...]
    lengths: tuple[int, ...]
    values: tuple[Number | None, ...]
    presence: tuple[int, ...]
    validity: tuple[int, ...]

    @property
    def size(self) -> int:
        return math.prod(self.lengths)

    def index_of(self, coordinate: Sequence[int]) -> int:
        return row_major_offset(coordinate, self.lengths)

    def state_at(self, coordinate: Sequence[int]) -> str:
        index = self.index_of(coordinate)
        if not self.presence[index]:
            return ABSENT
        return PRESENT_VALUE if self.validity[index] else PRESENT_NULL

    def value_at(self, coordinate: Sequence[int]) -> Number | None:
        return self.values[self.index_of(coordinate)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "lengths": list(self.lengths),
            "values": list(self.values),
            "presence": list(self.presence),
            "validity": list(self.validity),
        }


@dataclass(frozen=True)
class ManifestBlock:
    key: tuple[int, ...]
    shape: tuple[int, ...]
    stored_offsets: int
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "shape": list(self.shape),
            "stored_offsets": self.stored_offsets,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class SnapshotManifest:
    """快照清单：元数据、非空块描述与内容摘要。"""

    axes: tuple[str, ...]
    lengths: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    offset_order: str
    source_record_count: int
    cell_count: int
    block_count: int
    blocks: tuple[ManifestBlock, ...]
    digest: str
    logical_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "lengths": list(self.lengths),
            "chunk_shape": list(self.chunk_shape),
            "offset_order": self.offset_order,
            "source_record_count": self.source_record_count,
            "cell_count": self.cell_count,
            "block_count": self.block_count,
            "blocks": [block.as_dict() for block in self.blocks],
            "digest": self.digest,
            "logical_digest": self.logical_digest,
        }


# ---------------------------------------------------------------------------
# 命令行（只读展示，不写回原始资料）
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="稀疏坐标湖表：裁决、分块、窗口计划与清单（只读）"
    )
    parser.add_argument("coordinates", help="coordinates.jsonl 或同构记录文件")
    parser.add_argument("config", help="tensor.json 或同构配置")
    parser.add_argument("--axes", nargs="+", help="显式轴顺序；不一致时拒绝")
    parser.add_argument("--window", nargs="+", type=int, metavar="COORD",
                        help="可选的半开窗口，先给 lower 再给 upper，各 rank 个整数")
    parser.add_argument("--rechunk", nargs="+", type=int, help="可选的新 chunk_shape")
    parser.add_argument("--dense", action="store_true",
                        help="显式打印密集导出与 presence/validity 位图")
    args = parser.parse_args(argv)

    snapshot = load_snapshot(args.coordinates, args.config, expected_axes=args.axes)
    if args.window:
        low = args.window[: len(args.window) // 2]
        high = args.window[len(args.window) // 2 :]
        plan = snapshot.window_plan(low, high)
        result: dict[str, Any] = {
            "manifest": snapshot.manifest().as_dict(),
            "footprint": snapshot.footprint(),
            "window_plan": {
                "lower": list(plan.lower),
                "upper": list(plan.upper),
                "blocks": [
                    {
                        "key": list(block.key),
                        "effective_shape": list(block.effective_shape),
                        "intersection_lower": list(block.intersection_lower),
                        "intersection_upper": list(block.intersection_upper),
                        "stored_offsets": block.stored_offsets,
                    }
                    for block in plan.blocks
                ],
            },
            "window_cells": [
                {
                    "coordinate": list(cell.coordinate),
                    "state": cell.state,
                    "value": cell.value,
                    "revision": cell.revision,
                }
                for cell in snapshot.scan_window(low, high)
            ],
        }
    else:
        result = {
            "manifest": snapshot.manifest().as_dict(),
            "footprint": snapshot.footprint(),
        }
    if args.rechunk:
        rechunked = snapshot.rechunk(args.rechunk)
        result["rechunked"] = {
            "manifest": rechunked.manifest().as_dict(),
            "footprint": rechunked.footprint(),
            "logical_mapping_preserved": rechunked.logical_digest == snapshot.logical_digest,
        }
    if args.dense:
        result["dense_export"] = snapshot.dense_export().as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
