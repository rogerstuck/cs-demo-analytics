"""阶段 0：解析 CS2 demo 并导出结构化 Parquet 数据。

用法：
    python -m src.parser.parse <demo路径> [-o 输出目录] [--snapshot-seconds 1.0]

示例：
    python -m src.parser.parse "data/demos/my demo/xxx.dem" -o data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from awpy import Demo

# 非枪械/无法从快照推导的武器名（规整后保留，不做剔除）
EXTRA_WEAPON_NAMES: frozenset[str] = frozenset(
    {
        "world",  # 非玩家武器造成的死亡（坠亡/环境伤害）
        "knife",
        "knife_t",
        "knife_ct",
        "taser",
        "c4",
        "planted_c4",
        "hegrenade",
        "flashbang",
        "smokegrenade",
        "molotov",
        "incgrenade",
        "decoy",
        "inferno",
        "snowball",
        "shield",
    }
)


def canonical_weapon_names(snapshots: pl.DataFrame) -> set[str]:
    """汇总本 demo 里出现过的规范武器名。

    快照里的 ``primary_weapon`` / ``secondary_weapon`` 取自武器实体 → 物品定义
    索引，是**规范类名**（如 ``m4a1_silencer``），不会带涂装/名牌前缀。
    再补上投掷物、刀、``world`` 等非快照武器。
    """
    known: set[str] = set(EXTRA_WEAPON_NAMES)
    for col in ("primary_weapon", "secondary_weapon"):
        if col not in snapshots.columns:
            continue
        for v in snapshots[col].drop_nulls().unique().to_list():
            s = str(v).strip().lower()
            if s.startswith("weapon_"):
                s = s[len("weapon_") :]
            if s:
                known.add(s)
    return known


def normalize_weapon(raw: str | None, known: set[str]) -> str | None:
    """把原始武器字符串规整成规范类名。

    demo 的 ``player_death`` 等事件里，武器的字符串可能带上**涂装 / 名牌前缀**，
    例如同一把 M4A1-S 会出现两种写法：

    - ``m4a1_silencer``              （普通武器）
    - ``5e_2023pass1_m4a1_silencer`` （带 5E 平台 2023 赛季涂装前缀）

    这不是两种枪，只是服务器把外观名写进了事件字符串。本函数用**最长后缀匹配**
    把它还原成规范名，所以对任意平台前缀（5E / 完美 / Faceit …）都适用。

    Args:
        raw: 原始武器字符串（如 ``5e_2023pass1_ak47`` / ``weapon_ak47``）
        known: 规范武器名集合（见 :func:`canonical_weapon_names`）

    Returns:
        规范武器名；无法匹配时返回去掉 ``weapon_`` 前缀的小写原值。
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return s
    if s.startswith("weapon_"):
        s = s[len("weapon_") :]
    if s in known:
        return s
    parts = s.split("_")
    # 从最长的后缀开始尝试（i 越小后缀越长）
    for i in range(1, len(parts)):
        cand = "_".join(parts[i:])
        if cand in known:
            return cand
    return s


def add_weapon_clean(df: pl.DataFrame, known: set[str]) -> pl.DataFrame:
    """给表添加 ``weapon_clean`` 列（与 ``weapon`` 并列，原列不动）。"""
    if "weapon" not in df.columns:
        return df
    cleaned = [normalize_weapon(w, known) for w in df["weapon"].to_list()]
    return df.with_columns(pl.Series("weapon_clean", cleaned, dtype=pl.String))


def _safe_stdout() -> None:
    """让 Windows 控制台按 UTF-8 输出，避免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def annotate_players(players: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """标注 bot 与观战者实体。

    CS2 中 **bot 的 steamid 为 0**，GOTV 观战者实体（如 ``5EGOTV``）同样为 0，
    二者必须区分开，否则观战者会被误当成玩家。判定规则：

    - 观战者：steamid == 0 且（名字含 ``GOTV`` 或 side 为空）
    - bot：steamid == 0 且不是观战者

    Returns:
        (标注后的 players 表, bot 信息字典)
    """
    is_sid0 = pl.col("steamid") == 0
    name_up = pl.col("name").str.to_uppercase().fill_null("")
    is_spectator = is_sid0 & (name_up.str.contains("GOTV") | pl.col("side").is_null())

    df = players.with_columns(
        is_spectator.alias("is_spectator"),
        (is_sid0 & ~is_spectator).alias("is_bot"),
    )

    info = {
        "bot_steamids": sorted(df.filter(pl.col("is_bot"))["steamid"].unique().to_list()),
        "bot_names": sorted(df.filter(pl.col("is_bot"))["name"].unique().to_list()),
        "spectator_names": sorted(
            df.filter(pl.col("is_spectator"))["name"].unique().to_list()
        ),
        "note": (
            "CS2 中 bot 的 steamid 为 0，GOTV 观战者实体同样为 0。"
            "分析模块默认剔除 steamid == 0 的数据（即 bot 与观战者）。"
        ),
    }
    return df, info


def parse_demo(demo_path: str | Path, out_dir: str | Path, snapshot_seconds: float = 1.0) -> dict:
    """解析单个 demo，写出 Parquet 文件，返回汇总信息。"""
    demo_path = Path(demo_path).resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"找不到 demo 文件：{demo_path}")

    print(f"[1/3] 解析 demo：{demo_path.name}")
    demo = Demo(str(demo_path))
    demo.load("kills", "damages", "shots", "item_events")

    match_id = demo_path.stem
    out = Path(out_dir) / match_id
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] 写出到：{tables_dir}")

    # 元信息
    (tables_dir / "header.json").write_text(
        json.dumps(demo.header, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # players 表加上 bot / 观战者标记
    players, bot_info = annotate_players(demo.players)
    (tables_dir / "bots.json").write_text(
        json.dumps(bot_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if bot_info["bot_names"]:
        print(f"      检测到 bot：{', '.join(bot_info['bot_names'])}（steamid=0，分析时默认剔除）")

    snapshots = demo.snapshots(seconds=snapshot_seconds)
    known_weapons = canonical_weapon_names(snapshots)

    # 各数据集落库（Parquet 天然 UTF-8，中文名安全）
    tables: dict[str, object] = {
        "players": players,
        "stats": demo.stats,
        "rounds": demo.rounds,
        "kills": add_weapon_clean(demo.kills, known_weapons),
        "damages": add_weapon_clean(demo.damages, known_weapons),
        "shots": add_weapon_clean(demo.shots, known_weapons),
        "grenades": demo.grenades,
        "bomb": demo.bomb,
        "item_events": demo.item_events,
        "snapshots_1s": snapshots,
    }

    manifest: dict[str, dict] = {}
    for name, df in tables.items():
        path = tables_dir / f"{name}.parquet"
        df.write_parquet(path)
        manifest[name] = {"rows": df.shape[0], "columns": df.shape[1]}

    (tables_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "demo": str(demo_path),
        "map": demo.header.get("map_name"),
        "match_id": match_id,
        "output_dir": str(out),
        "tables": manifest,
        "bots": bot_info["bot_names"],
    }
    return summary


def main() -> int:
    _safe_stdout()
    parser = argparse.ArgumentParser(description="解析 CS2 demo 为 Parquet 数据")
    parser.add_argument("demos", nargs="+", help="demo 文件路径（可多个）")
    parser.add_argument("-o", "--out", default="data/processed", help="输出目录")
    parser.add_argument(
        "--snapshot-seconds", type=float, default=1.0, help="坐标快照采样间隔（秒）"
    )
    args = parser.parse_args()

    for d in args.demos:
        summary = parse_demo(d, args.out, args.snapshot_seconds)
        print("\n[3/3] 完成 ✅")
        print(f"  地图：{summary['map']}")
        print(f"  输出：{summary['output_dir']}")
        print("  各表行数：")
        for name, info in summary["tables"].items():
            print(f"    {name:<16} {info['rows']:>8} 行 × {info['columns']} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
