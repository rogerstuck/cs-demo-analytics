"""分析模块共用的读表与清洗工具。

约定：
- 解析产物位于 ``data/processed/<demo名>/tables/*.parquet``
- **bot 与 GOTV 观战者的 steamid 都是 0**，分析时默认剔除（见 `exclude_bots`）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

# CS2 的"长枪"（步枪 / 狙击 / 机枪）——用于判断这一回合是否买够了
RIFLES: frozenset[str] = frozenset(
    {
        "ak47",
        "m4a1",
        "m4a4",
        "m4a1_silencer",
        "awp",
        "ssg08",
        "aug",
        "sg556",
        "famas",
        "galilar",
        "g3sg1",
        "scar20",
        "m249",
        "negev",
    }
)

SIDE_SHORT = {"counter-terrorist": "CT", "terrorist": "T"}


def safe_stdout() -> None:
    """让 Windows 控制台按 UTF-8 输出，避免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_map_name(data_dir: str | Path) -> str:
    """从 header.json 读取地图名。"""
    header = json.loads((Path(data_dir) / "tables" / "header.json").read_text(encoding="utf-8"))
    return header["map_name"]


def load_table(data_dir: str | Path, name: str, *, exclude_bots: bool = True) -> pl.DataFrame:
    """读取一张解析表。

    Args:
        data_dir: ``data/processed/<demo名>``
        name: 表名（不含 ``.parquet``）
        exclude_bots: 是否剔除 steamid == 0 的行（bot 与观战者）
    """
    df = pl.read_parquet(Path(data_dir) / "tables" / f"{name}.parquet")
    return drop_bots(df) if exclude_bots else df


def drop_bots(df: pl.DataFrame) -> pl.DataFrame:
    """剔除 ``steamid == 0`` 的行（bot 与 GOTV 观战者）。

    表没有 ``steamid`` 列时原样返回。
    """
    if "steamid" not in df.columns:
        return df
    return df.filter(pl.col("steamid") != 0)


def freeze_end_snapshot(snapshots: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    """取每个回合 freeze 结束时刻（买入完成、未开局）的最后一份快照。

    快照是抽样数据（默认 1 秒一帧），所以取 ``start_tick <= tick <= freeze_end_tick``
    范围内的最后一行，并附上 ``round_num``。
    """
    picks: list[pl.DataFrame] = []
    for r in rounds.iter_rows(named=True):
        g = snapshots.filter(
            (pl.col("tick") >= r["start_tick"]) & (pl.col("tick") <= r["freeze_end_tick"])
        )
        if g.height == 0:
            continue
        g = g.filter(pl.col("tick") == g["tick"].max())
        picks.append(g.with_columns(pl.lit(r["round_num"]).cast(pl.Int32).alias("round_num")))
    if not picks:
        return snapshots.head(0).with_columns(
            pl.lit(None, dtype=pl.Int32).alias("round_num")
        )
    return pl.concat(picks, how="vertical_relaxed")


def assign_round_numbers(
    df: pl.DataFrame, rounds: pl.DataFrame, *, tick_col: str = "tick"
) -> pl.DataFrame:
    """按 tick 把事件/快照归到所属回合（新增 ``round_num`` 列）。"""
    r = rounds.sort("start_tick").select(["round_num", "start_tick"])
    return (
        df.sort(tick_col)
        .join_asof(r, left_on=tick_col, right_on="start_tick", strategy="backward")
        .filter(pl.col(tick_col) <= rounds["end_tick"].max())
    )


def pistol_round_numbers(snapshots: pl.DataFrame, rounds: pl.DataFrame) -> list[int]:
    """判定哪些回合是手枪局（= 每个半场的第一回合）。

    判据：demo 第一回合 + **出现大规模换边的回合**。

    Note:
        不能用"全员带入装备值只剩免费手枪（<=200）"来判定：**纯 eco 局**
        （上回合没人带枪回来）同样满足这个条件，会被误判成手枪局。
        而换边（双方阵营互换）是半场开始的唯一可靠信号。
    """
    order = rounds.sort("start_tick")["round_num"].to_list()
    if not order:
        return []

    snap = assign_round_numbers(
        snapshots.select(["tick", "steamid", "side"]), rounds
    ).filter(pl.col("steamid") != 0)
    if snap.height == 0:
        return [int(order[0])]

    sides: dict[int, dict[int, str]] = {}
    for (rn,), g in snap.group_by(["round_num"], maintain_order=True):
        if rn is None:
            continue
        first = g.group_by("steamid").agg(pl.col("side").first())
        sides[int(rn)] = dict(zip(first["steamid"].to_list(), first["side"].to_list()))

    pistols = {int(order[0])}
    prev = sides.get(int(order[0]), {})
    for rn in order[1:]:
        cur = sides.get(int(rn), {})
        common = set(prev) & set(cur)
        if common:
            flipped = sum(1 for sid in common if prev[sid] != cur[sid])
            if flipped > len(common) / 2:
                pistols.add(int(rn))
        if cur:
            prev = cur
    return sorted(pistols)
