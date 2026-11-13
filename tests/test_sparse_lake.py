import hashlib
import json
import unittest
from pathlib import Path

from src.sparse_lake import (
    ABSENT,
    PRESENT_NULL,
    PRESENT_VALUE,
    AxisOrderError,
    Block,
    BlockLocation,
    InvalidCoordinateError,
    RevisionConflictError,
    Snapshot,
    SnapshotMeta,
    build_snapshot,
    chunk_effective_shape,
    chunk_key,
    chunk_local,
    load_snapshot,
    resolve_records,
    row_major_offset,
    snapshot_from_config,
    unflatten_offset,
)

ROOT = Path(__file__).resolve().parents[1]
META = SnapshotMeta(axes=("device", "tick"), lengths=(3, 5), chunk_shape=(2, 3))


def record(coordinate, value, revision=1, tombstone=False):
    return {
        "coordinate": list(coordinate),
        "value": value,
        "revision": revision,
        "tombstone": tombstone,
    }


class EncodingContractTests(unittest.TestCase):
    def test_documented_edge_block_example(self):
        coordinate = (2, 4)
        self.assertEqual(chunk_key(META, coordinate), (1, 1))
        self.assertEqual(chunk_local(META, coordinate), (0, 1))
        self.assertEqual(chunk_effective_shape(META, (1, 1)), (1, 2))
        self.assertEqual(row_major_offset((0, 1), (1, 2)), 1)

    def test_full_blocks_use_full_chunk_shape(self):
        self.assertEqual(chunk_effective_shape(META, (0, 0)), (2, 3))
        self.assertEqual(chunk_effective_shape(META, (1, 0)), (1, 3))
        self.assertEqual(row_major_offset((1, 2), (2, 3)), 5)

    def test_offsets_round_trip_for_every_block_shape(self):
        for key in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            shape = chunk_effective_shape(META, key)
            for offset in range(shape[0] * shape[1]):
                local = unflatten_offset(offset, shape)
                self.assertTrue(all(0 <= local[i] < shape[i] for i in range(2)))
                self.assertEqual(row_major_offset(local, shape), offset)

    def test_out_of_range_coordinate_rejected(self):
        for coordinate in [(-1, 0), (0, 5), (3, 0), (0, -2)]:
            with self.subTest(coordinate=coordinate), self.assertRaises(InvalidCoordinateError):
                chunk_key(META, coordinate)

    def test_out_of_range_offset_rejected(self):
        with self.assertRaises(InvalidCoordinateError):
            unflatten_offset(2, (1, 2))
        with self.assertRaises(InvalidCoordinateError):
            row_major_offset((1, 0), (1, 2))

    def test_padding_is_not_stored_as_absence(self):
        snapshot = build_snapshot([record((2, 4), 8)], META)
        block = snapshot.blocks[(1, 1)]
        self.assertEqual(block.shape, (1, 2))
        self.assertEqual(block.offsets, (1,))
        self.assertNotIn(0, block.offsets)
        # 块 (1,1) 有效区域仅 1 行 2 列，没有越界 padding 槽位。
        self.assertEqual(block.shape[0] * block.shape[1], 2)


class ResolutionTests(unittest.TestCase):
    def test_duplicate_revision_uses_highest(self):
        cells = resolve_records(
            [record((2, 4), 7, 1), record((2, 4), 8, 2)], META
        )
        self.assertEqual(len(cells), 1)
        self.assertEqual((cells[0].state, cells[0].value, cells[0].revision), (PRESENT_VALUE, 8, 2))

    def test_tie_on_highest_revision_rejected(self):
        with self.assertRaises(RevisionConflictError):
            build_snapshot([record((2, 4), 7, 2), record((2, 4), 8, 2)], META)

    def test_older_tie_below_top_revision_does_not_conflict(self):
        # revision 1 上两条记录平级，但 revision 2 是唯一最高者，裁决只看最高 revision。
        snapshot = build_snapshot(
            [
                record((2, 4), 7, 1),
                record((2, 4), 8, 1),
                record((2, 4), 9, 2),
            ],
            META,
        )
        self.assertEqual(snapshot.restore((2, 4)).value, 9)
        self.assertEqual(snapshot.restore((2, 4)).revision, 2)

    def test_tombstone_removes_coordinate(self):
        cells = resolve_records(
            [record((0, 0), 0, 1), record((0, 0), None, 2, tombstone=True)], META
        )
        self.assertEqual(cells, ())

    def test_older_tombstone_loses_to_newer_value(self):
        cells = resolve_records(
            [record((0, 0), None, 1, tombstone=True), record((0, 0), 4, 2)], META
        )
        self.assertEqual(cells[0].state, PRESENT_VALUE)
        self.assertEqual(cells[0].value, 4)

    def test_tombstone_requires_explicit_null(self):
        with self.assertRaises(Exception):
            build_snapshot([record((0, 0), 5, 2, tombstone=True)], META)

    def test_tombstone_does_not_erase_earlier_state_with_lower_revision(self):
        snapshot = build_snapshot(
            [record((0, 0), 3, 1), record((0, 0), None, 2, tombstone=True)], META
        )
        self.assertEqual(snapshot.cell_count, 0)

    def test_duplicates_resolved_before_chunking(self):
        snapshot = build_snapshot(
            [record((2, 4), 7, 1), record((2, 4), 8, 2)], META
        )
        block = snapshot.blocks[(1, 1)]
        self.assertEqual(block.values, (8,))
        self.assertEqual(len(snapshot.index), 1)


class ThreeStateTests(unittest.TestCase):
    def test_zero_null_and_absent_are_distinguishable(self):
        snapshot = load_snapshot(
            ROOT / "fixtures/coordinates.jsonl",
            ROOT / "fixtures/tensor.json",
            expected_axes=("device", "tick"),
        )
        self.assertEqual(snapshot.restore((0, 0)).state, PRESENT_VALUE)
        self.assertEqual(snapshot.restore((0, 0)).value, 0)
        self.assertEqual(snapshot.restore((0, 1)).state, PRESENT_NULL)
        self.assertIsNone(snapshot.restore((0, 1)).value)
        self.assertEqual(snapshot.restore((0, 2)).state, ABSENT)
        self.assertIsNone(snapshot.restore((0, 2)).revision)

    def test_negative_values_supported_and_not_filled(self):
        snapshot = build_snapshot(
            [record((1, 2), -4.5), record((1, 3), -0.0)], META
        )
        self.assertEqual(snapshot.restore((1, 2)).value, -4.5)
        self.assertEqual(snapshot.restore((1, 3)).value, -0.0)
        self.assertEqual(snapshot.restore((1, 4)).state, ABSENT)

    def test_nan_and_infinity_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaises(Exception):
                build_snapshot([record((0, 0), bad)], META)

    def test_bool_values_rejected(self):
        with self.assertRaises(Exception):
            build_snapshot([record((0, 0), True)], META)

    def test_invalid_coordinate_in_record_rejected(self):
        with self.assertRaises(InvalidCoordinateError):
            build_snapshot([record((3, 5), 1)], META)


class SnapshotStructureTests(unittest.TestCase):
    def test_only_nonempty_blocks_stored(self):
        snapshot = build_snapshot([record((2, 4), 8)], META)
        self.assertEqual(list(snapshot.blocks), [(1, 1)])
        self.assertEqual(snapshot.blocks[(1, 1)].offsets, (1,))
        self.assertEqual(snapshot.block_count, 1)
        self.assertNotIn((0, 0), snapshot.blocks)

    def test_empty_block_rejected(self):
        with self.assertRaises(Exception):
            Block(key=(0, 0), shape=(2, 3), offsets=(), values=(), revisions=())

    def test_manifest_contains_effective_shapes_and_digests(self):
        snapshot = build_snapshot(
            [record((0, 0), 0), record((0, 1), None), record((2, 4), 8)], META
        )
        manifest = snapshot.manifest()
        self.assertEqual(manifest.source_record_count, 3)
        self.assertEqual(manifest.cell_count, 3)
        self.assertEqual(manifest.block_count, 2)
        shapes = {tuple(block.key): block.shape for block in manifest.blocks}
        self.assertEqual(shapes, {(0, 0): (2, 3), (1, 1): (1, 2)})
        self.assertNotEqual(manifest.digest, manifest.logical_digest)

    def test_snapshot_is_immutable(self):
        snapshot = build_snapshot([record((0, 0), 0)], META)
        with self.assertRaises(AttributeError):
            snapshot._meta = META
        with self.assertRaises(TypeError):
            snapshot.blocks[(9, 9)] = None

    def test_inconsistent_index_rejected(self):
        block = Block(key=(1, 1), shape=(1, 2), offsets=(1,), values=(8,), revisions=(2,))
        with self.assertRaises(Exception):
            Snapshot(
                META,
                {(1, 1): block},
                {(2, 3): BlockLocation((1, 1), 1)},
                1,
            )

    def test_dangling_offset_index_rejected(self):
        block = Block(key=(0, 0), shape=(2, 3), offsets=(0,), values=(0,), revisions=(1,))
        # 坐标 (0, 1) 解码后与索引一致，但块并未占用其 offset。
        with self.assertRaises(Exception):
            Snapshot(META, {(0, 0): block}, {(0, 1): BlockLocation((0, 0), 1)}, 1)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = build_snapshot(
            [
                record((0, 0), 0),
                record((0, 1), None),
                record((1, 4), 5),
                record((2, 4), 8),
            ],
            META,
        )

    def test_window_is_half_open(self):
        rows = self.snapshot.scan_window((0, 0), (1, 1))
        self.assertEqual([cell.coordinate for cell in rows], [(0, 0)])

    def test_window_plan_reports_intersections_only(self):
        plan = self.snapshot.window_plan((1, 2), (3, 5))
        # 三个非空块的几何交集都与窗口相交；并不枚举 2x4 的完整块网格。
        self.assertEqual(plan.block_count, 3)
        by_key = {block.key: block for block in plan.blocks}
        self.assertEqual(by_key[(0, 0)].intersection_lower, (1, 2))
        self.assertEqual(by_key[(0, 0)].intersection_upper, (2, 3))
        self.assertEqual(by_key[(0, 0)].effective_shape, (2, 3))
        self.assertEqual(by_key[(0, 1)].intersection_lower, (1, 3))
        self.assertEqual(by_key[(0, 1)].intersection_upper, (2, 5))
        self.assertEqual(by_key[(0, 1)].effective_shape, (2, 2))
        self.assertEqual(by_key[(1, 1)].intersection_lower, (2, 3))
        self.assertEqual(by_key[(1, 1)].intersection_upper, (3, 5))
        self.assertEqual(by_key[(1, 1)].effective_shape, (1, 2))

    def test_window_does_not_create_empty_block_entries(self):
        plan = self.snapshot.window_plan((0, 0), (3, 5))
        self.assertEqual(plan.block_count, self.snapshot.block_count)

    def test_empty_window(self):
        self.assertEqual(self.snapshot.scan_window((1, 1), (1, 1)), ())

    def test_invalid_window_rejected(self):
        for lower, upper in [((0, 0), (3, 6)), ((-1, 0), (1, 1)), ((2, 0), (1, 1))]:
            with self.subTest(lower=lower, upper=upper), self.assertRaises(InvalidCoordinateError):
                self.snapshot.window_plan(lower, upper)

    def test_axes_order_in_window_rejected(self):
        with self.assertRaises(AxisOrderError):
            self.snapshot.window_plan((0, 0), (3, 5), axes=("tick", "device"))


class AxisOrderTests(unittest.TestCase):
    def test_config_axis_order_checked(self):
        config = json.loads((ROOT / "fixtures/tensor.json").read_text(encoding="utf-8"))
        records = json.loads(
            "[{\"coordinate\":[0,0],\"value\":0,\"revision\":1,\"tombstone\":false}]"
        )
        self.assertIsNotNone(snapshot_from_config(records, config, expected_axes=("device", "tick")))
        with self.assertRaises(AxisOrderError):
            snapshot_from_config(records, config, expected_axes=("tick", "device"))

    def test_rechunk_rejects_swapped_axes(self):
        snapshot = build_snapshot([record((0, 0), 0)], META)
        with self.assertRaises(AxisOrderError):
            snapshot.rechunk((1, 2), axes=("tick", "device"))

    def test_restore_rejects_swapped_axes(self):
        snapshot = build_snapshot([record((0, 0), 0)], META)
        with self.assertRaises(AxisOrderError):
            snapshot.restore((0, 0), axes=("tick", "device"))


class RechunkTests(unittest.TestCase):
    def test_rechunk_preserves_logical_mapping(self):
        snapshot = build_snapshot(
            [
                record((0, 0), 0),
                record((0, 1), None),
                record((1, 2), -3),
                record((2, 4), 8),
            ],
            META,
        )
        rechunked = snapshot.rechunk((1, 2))
        self.assertEqual(rechunked.logical_digest, snapshot.logical_digest)
        self.assertEqual(rechunked.cell_count, snapshot.cell_count)
        self.assertEqual(rechunked.source_record_count, snapshot.source_record_count)
        for coordinate in [(0, 0), (0, 1), (1, 2), (2, 4), (0, 2), (1, 1)]:
            before = snapshot.restore(coordinate)
            after = rechunked.restore(coordinate)
            self.assertEqual((before.state, before.value, before.revision), (after.state, after.value, after.revision))
        self.assertNotEqual(rechunked.digest, snapshot.digest)

    def test_rechunk_not_just_equal_sums(self):
        # 同一坐标上的 0 与 NULL 在求和意义下都是“空”，但重分块必须保留三态。
        snapshot = build_snapshot([record((1, 2), 0), record((1, 3), None)], META)
        rechunked = snapshot.rechunk((3, 1))
        self.assertEqual(snapshot.footprint()["present_value_cells"], 1)
        self.assertEqual(rechunked.footprint()["present_null_cells"], 1)
        self.assertEqual(rechunked.restore((1, 2)).state, PRESENT_VALUE)
        self.assertEqual(rechunked.restore((1, 2)).value, 0)
        self.assertEqual(rechunked.restore((1, 3)).state, PRESENT_NULL)

    def test_rechunk_to_full_shape_keeps_edge_semantics(self):
        snapshot = build_snapshot([record((2, 4), 8)], META)
        rechunked = snapshot.rechunk((3, 5))
        self.assertEqual(list(rechunked.blocks), [(0, 0)])
        self.assertEqual(rechunked.blocks[(0, 0)].shape, (3, 5))
        self.assertEqual(rechunked.blocks[(0, 0)].offsets, (14,))
        self.assertEqual(rechunked.restore((2, 4)).value, 8)

    def test_rechunk_rejects_bad_chunk_shape(self):
        snapshot = build_snapshot([record((0, 0), 0)], META)
        for chunk in [(0, 2), (2,), (2, -1)]:
            with self.subTest(chunk=chunk), self.assertRaises(Exception):
                snapshot.rechunk(chunk)


class FullMappingTests(unittest.TestCase):
    def test_all_coordinates_match_bruteforce_oracle(self):
        meta = SnapshotMeta(axes=("a", "b", "c"), lengths=(3, 4, 2), chunk_shape=(2, 3, 1))
        records = []
        expected = {}
        revision = 0
        for a in range(3):
            for b in range(4):
                for c in range(2):
                    coordinate = (a, b, c)
                    # 决定每格是缺测、显式 NULL 还是数值（含 0 与负数）。
                    mode = (a + 2 * b + 3 * c) % 4
                    if mode == 0:
                        continue
                    revision += 1
                    value = None if mode == 1 else (0 if mode == 2 else -revision)
                    records.append(record(coordinate, value, revision))
                    expected[coordinate] = (PRESENT_NULL if value is None else PRESENT_VALUE, value)
        snapshot = build_snapshot(records, meta)
        self.assertEqual(snapshot.dense_export().size, 24)
        for coordinate in [
            (a, b, c) for a in range(3) for b in range(4) for c in range(2)
        ]:
            cell = snapshot.restore(coordinate)
            if coordinate in expected:
                state, value = expected[coordinate]
            else:
                state, value = ABSENT, None
            self.assertEqual((cell.state, cell.value), (state, value), coordinate)
            if state == ABSENT:
                self.assertIsNone(cell.revision)
            else:
                self.assertIsNotNone(cell.revision)
        exported = snapshot.dense_export()
        for coordinate in expected:
            self.assertEqual(exported.state_at(coordinate), expected[coordinate][0])

    def test_window_matches_full_scan_for_rand_rectangles(self):
        meta = SnapshotMeta(axes=("x", "y"), lengths=(5, 7), chunk_shape=(2, 3))
        snapshot = build_snapshot(
            [record((x, y), (x * 7 + y) % 5, revision=1) for x in range(5) for y in range(7) if (x + y) % 3 != 0],
            meta,
        )
        for lower in [(0, 0), (1, 2), (3, 5)]:
            for upper in [(2, 3), (4, 6), (5, 7)]:
                if not all(lo <= hi for lo, hi in zip(lower, upper)):
                    continue
                window = snapshot.scan_window(lower, upper)
                direct = [
                    snapshot.restore((x, y))
                    for x in range(lower[0], upper[0])
                    for y in range(lower[1], upper[1])
                    if snapshot.restore((x, y)).state != ABSENT
                ]
                self.assertEqual(
                    [(cell.coordinate, cell.state, cell.value) for cell in window],
                    [(cell.coordinate, cell.state, cell.value) for cell in direct],
                )


class DenseExportTests(unittest.TestCase):
    def test_presence_and_validity_distinguish_three_states(self):
        snapshot = build_snapshot(
            [record((0, 0), 0), record((0, 1), None), record((2, 4), 8)], META
        )
        exported = snapshot.dense_export()
        self.assertEqual(exported.state_at((0, 0)), PRESENT_VALUE)
        self.assertEqual(exported.state_at((0, 1)), PRESENT_NULL)
        self.assertEqual(exported.state_at((0, 2)), ABSENT)
        self.assertEqual(exported.value_at((0, 0)), 0)
        self.assertIsNone(exported.value_at((0, 1)))
        self.assertIsNone(exported.value_at((0, 2)))
        i00, i01, i02 = (exported.index_of(c) for c in [(0, 0), (0, 1), (0, 2)])
        self.assertEqual((exported.presence[i00], exported.validity[i00]), (1, 1))
        self.assertEqual((exported.presence[i01], exported.validity[i01]), (1, 0))
        self.assertEqual((exported.presence[i02], exported.validity[i02]), (0, 0))
        # 未记录与 NULL 的 values 都是 None，靠位图区分，绝不自动填 0。
        self.assertEqual(exported.values[i02], None)

    def test_dense_export_is_explicit_opt_in(self):
        snapshot = build_snapshot([record((0, 0), 1)], META)
        self.assertFalse(hasattr(snapshot, "array"))
        self.assertEqual(snapshot.footprint()["recorded_cells"], 1)
        self.assertEqual(snapshot.footprint()["block_grid_cells"], 4)
        self.assertLess(snapshot.block_count, snapshot.footprint()["block_grid_cells"])


class SparseFootprintTests(unittest.TestCase):
    def test_large_sparse_domain_does_not_materialize_grid(self):
        meta = SnapshotMeta(
            axes=("device", "tick", "probe"),
            lengths=(2_000_000, 10_000, 64),
            chunk_shape=(1_000, 100, 8),
        )
        records = [
            record((0, 0, 0), 0),
            record((1_999_999, 9_999, 63), None),
            record((1_500_000, 7_500, 32), -1),
        ]
        snapshot = build_snapshot(records, meta)
        footprint = snapshot.footprint()
        logical = 2_000_000 * 10_000 * 64
        self.assertEqual(footprint["logical_cells"], logical)
        self.assertEqual(footprint["recorded_cells"], 3)
        self.assertEqual(footprint["nonempty_blocks"], 3)
        self.assertLess(footprint["nonempty_blocks"], footprint["block_grid_cells"])
        self.assertEqual(footprint["stored_entries"], 3)
        self.assertEqual(len(snapshot.blocks), 3)
        self.assertLess(len(snapshot.index), 4)

    def test_manifest_does_not_list_all_blocks(self):
        meta = SnapshotMeta(axes=("a", "b"), lengths=(100, 100), chunk_shape=(1, 1))
        snapshot = build_snapshot([record((0, 0), 0), record((99, 99), None)], meta)
        manifest = snapshot.manifest()
        self.assertEqual(len(manifest.blocks), 2)
        self.assertLess(manifest.block_count, 10_000)


class FixtureIntegrityTests(unittest.TestCase):
    def test_inputs_not_modified_by_lake(self):
        inventory = json.loads((ROOT / "material-index.json").read_text(encoding="utf-8"))
        for item in inventory:
            with self.subTest(path=item["path"]):
                digest = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
                self.assertEqual(digest, item["sha256"])
                load_snapshot(
                    ROOT / "fixtures/coordinates.jsonl",
                    ROOT / "fixtures/tensor.json",
                    expected_axes=("device", "tick"),
                )
                self.assertEqual(
                    hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(),
                    item["sha256"],
                )

    def test_manifest_fixture_values(self):
        snapshot = load_snapshot(
            ROOT / "fixtures/coordinates.jsonl",
            ROOT / "fixtures/tensor.json",
        )
        footprint = snapshot.footprint()
        self.assertEqual(footprint["recorded_cells"], 3)
        self.assertEqual(footprint["present_value_cells"], 2)
        self.assertEqual(footprint["present_null_cells"], 1)
        self.assertEqual(footprint["absent_cells"], 12)
        self.assertEqual(snapshot.restore((2, 4)).value, 8)


if __name__ == "__main__":
    unittest.main()
