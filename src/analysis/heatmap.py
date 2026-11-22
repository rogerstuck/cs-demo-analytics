"""阶段 1：位置 / 交火热力图。

基于已解析的 Parquet 数据，用 awpy 自带的雷达图 + 坐标变换生成热力图。

用法：
    # 全员走位热力图
    python -m src.analysis.heatmap --data data/processed/<match_id>

    # 只看 CT / T
    python -m src.analysis.heatmap --data data/processed/<match_id> --side CT
    python -m src.analysis.heatmap --data data/processed/<match_id> --side T

    # 只看某个玩家（支持昵称片段匹配）
    python -m src.analysis.heatmap --data data/processed/<match_id> --player 牛牛

    # 死亡热区（kills 表的被击杀者坐标）
    python -m src.analysis.heatmap --data data/processed/<match_id> --table kills --role victim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，便于命令行保存图片

import polars as pl
from awpy.plot import heatmap as draw_heatmap

from .common import drop_bots, load_map_name

_SIDE_MAP = {
    "t": "terrorist",
    "terrorist": "terrorist",
    "ct": "counter-terrorist",
    "counter-terrorist": "counter-terrorist",
    "counter_terrorist": "counter-terrorist",
}


def load_points(
    data_dir: Path,
    table: str = "snapshots_1s",
    *,
    side: str | None = None,
    player: str | None = None,
    role: str | None = None,
) -> list[tuple[float, float]]:
    """从指定表读取 (x, y) 坐标点，支持阵营/玩家过滤。

    bot 与观战者（steamid == 0）默认剔除。
    """
    df = pl.read_parquet(data_dir / "tables" / f"{table}.parquet")

    if table == "snapshots_1s":
        x_col, y_col = "x", "y"
        side_col, name_col = "side", "name"
        df = drop_bots(df)
    elif table == "kills":
        if role not in ("attacker", "victim"):
            raise ValueError("table=kills 时必须指定 --role attacker 或 victim")
        x_col, y_col = f"{role}_x", f"{role}_y"
        side_col, name_col = f"{role}_side", f"{role}_name"
        df = df.filter(pl.col(f"{role}_steamid") != 0)
    else:
        raise ValueError(f"暂不支持的表：{table}（可选 snapshots_1s / kills）")

    if side:
        side = _SIDE_MAP.get(side.lower(), side)
        df = df.filter(pl.col(side_col) == side)
    if player:
        df = df.filter(pl.col(name_col).str.contains(player))

    return list(zip(df[x_col].to_list(), df[y_col].to_list()))


def make_heatmap(
    data_dir: Path,
    *,
    table: str = "snapshots_1s",
    side: str | None = None,
    player: str | None = None,
    role: str | None = None,
    method: str = "kde",
    bins: int = 48,
    out: Path | None = None,
) -> Path:
    """生成热力图，返回保存路径。"""
    map_name = load_map_name(data_dir)
    if side:
        side = _SIDE_MAP.get(side.lower(), side)
    points = load_points(data_dir, table, side=side, player=player, role=role)
    if not points:
        raise SystemExit("没有符合条件的数据点，请检查过滤条件")

    fig, ax = draw_heatmap(map_name, points, method=method, bins=bins, alpha=0.75)

    # 标题
    parts = [map_name, table]
    if side:
        parts.append("CT" if side == "counter-terrorist" else "T")
    if player:
        parts.append(player)
    if role:
        parts.append(role)
    ax.set_title(" - ".join(parts) + f"  (n={len(points)})", color="white", pad=8)

    if out is None:
        tag = "-".join(parts)
        figures_dir = data_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        out = figures_dir / f"heatmap_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="生成 CS2 位置/交火热力图")
    parser.add_argument("--data", required=True, help="已解析数据的目录（含 *.parquet 和 header.json）")
    parser.add_argument("--table", default="snapshots_1s", help="数据表：snapshots_1s（走位）或 kills（交火）")
    parser.add_argument("--side", help="阵营过滤：T / CT")
    parser.add_argument("--player", help="玩家昵称（片段匹配）")
    parser.add_argument("--role", help="kills 表专用：attacker（击杀者）/ victim（被击杀者）")
    parser.add_argument("--method", default="kde", choices=["hex", "hist", "kde"], help="密度算法")
    parser.add_argument("--bins", type=int, default=48, help="分箱分辨率")
    parser.add_argument("--out", help="输出图片路径（默认自动命名到数据目录）")
    args = parser.parse_args()

    out = make_heatmap(
        Path(args.data),
        table=args.table,
        side=args.side,
        player=args.player,
        role=args.role,
        method=args.method,
        bins=args.bins,
        out=Path(args.out) if args.out else None,
    )
    print(f"[OK] 已保存热力图：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
