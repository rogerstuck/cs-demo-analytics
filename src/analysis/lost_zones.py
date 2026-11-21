"""阶段 2：失守区域分析。

定位"经常先掉人、且先掉人后容易输回合"的区域（失守/危险区）。

核心指标：
- 首死数 opening_deaths：每回合第一个阵亡（opening kill 的受害者）发生在该区域的次数
- 丢分率 loss_rate：该区域先掉首死后，先掉人一方输掉回合的比例

用法：
    python -m src.analysis.lost_zones --data data/processed/<match_id> [--grid 12] [--min-samples 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Windows 中文字体，避免图片标题变方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import polars as pl
from awpy.plot import pixel_to_world, radar, world_to_pixel

RADAR = 1024  # awpy 雷达图为 1024x1024


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_map_name(data_dir: Path) -> str:
    header = json.loads((data_dir / "header.json").read_text(encoding="utf-8"))
    return header["map_name"]


def assign_rounds(kills: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    """按 tick 把每次击杀归到对应回合。"""
    rounds_sorted = rounds.sort("start_tick").select(
        ["round_num", "start_tick", "end_tick", "winner_side"]
    )
    joined = kills.sort("tick").join_asof(
        rounds_sorted,
        left_on="tick",
        right_on="start_tick",
        strategy="backward",
    )
    return joined.filter(pl.col("tick") <= pl.col("end_tick"))


def opening_kills(kills_with_round: pl.DataFrame) -> pl.DataFrame:
    """每个回合的第一击杀（opening kill）。"""
    return kills_with_round.group_by("round_num").agg(pl.all().sort_by("tick").first())


def pixel_to_cell(px: float, py: float, n: int) -> tuple[int, int]:
    """雷达像素坐标 -> 网格 (row, col)。"""
    c = min(n - 1, max(0, int(px / (RADAR / n))))
    r = min(n - 1, max(0, int(py / (RADAR / n))))
    return r, c


def analyze(data_dir: Path, grid: int = 12, min_samples: int = 2) -> dict:
    """计算每个 zone 的首死数与丢分率，返回结构化结果。"""
    map_name = load_map_name(data_dir)
    kills = pl.read_parquet(data_dir / "kills.parquet")
    rounds = pl.read_parquet(data_dir / "rounds.parquet")

    k = assign_rounds(kills, rounds)
    opening = opening_kills(k)

    # ---- 首死统计 ----
    opening_deaths: dict[tuple[int, int], int] = defaultdict(int)  # (r,c) -> 首死数
    opening_by_side: dict[tuple[str, int, int], int] = defaultdict(int)
    opening_loss: dict[tuple[int, int], int] = defaultdict(int)  # (r,c) -> 先掉人后输掉回合数

    for row in opening.iter_rows(named=True):
        px, py = world_to_pixel(map_name, (row["victim_x"], row["victim_y"]))
        r, c = pixel_to_cell(px, py, grid)
        opening_deaths[(r, c)] += 1
        opening_by_side[(row["victim_side"], r, c)] += 1
        if row["winner_side"] != row["victim_side"]:
            opening_loss[(r, c)] += 1

    # ---- 全部死亡统计（危险区：哪里死人多）----
    all_deaths: dict[tuple[int, int], int] = defaultdict(int)
    for row in kills.iter_rows(named=True):
        px, py = world_to_pixel(map_name, (row["victim_x"], row["victim_y"]))
        r, c = pixel_to_cell(px, py, grid)
        all_deaths[(r, c)] += 1

    # ---- 汇总成表 ----
    cell_size = RADAR / grid
    records = []
    for (r, c), od in opening_deaths.items():
        loss = opening_loss[(r, c)]
        # 世界坐标中心（便于对照地图）
        px_center = (c + 0.5) * cell_size
        py_center = (r + 0.5) * cell_size
        wx, wy = pixel_to_world(map_name, (px_center, py_center))
        records.append(
            {
                "row": r,
                "col": c,
                "opening_deaths": od,
                "losses": loss,
                "loss_rate": round(loss / od, 2) if od else 0.0,
                "total_deaths": all_deaths[(r, c)],
                "ct_first_deaths": opening_by_side.get(("counter-terrorist", r, c), 0),
                "t_first_deaths": opening_by_side.get(("terrorist", r, c), 0),
                "world_x": round(wx),
                "world_y": round(wy),
            }
        )

    return {
        "map_name": map_name,
        "grid": grid,
        "records": records,
        "opening_deaths": opening_deaths,
        "opening_loss": opening_loss,
        "all_deaths": all_deaths,
    }


def build_grid(n: int, cell_values: dict[tuple[int, int], float]) -> np.ndarray:
    """把 (r,c)->值 转成 n×n 的 2D 数组（NaN 表示无数据）。"""
    g = np.full((n, n), np.nan)
    for (r, c), v in cell_values.items():
        g[r, c] = v
    return g


def draw_overlay(
    map_name: str, n: int, cell_values: dict[tuple[int, int], float], *, cmap: str, title: str, out: Path
) -> None:
    """在雷达图上叠加 zone 级网格着色。"""
    fig, ax = radar(map_name)
    grid = build_grid(n, cell_values)
    masked = np.ma.masked_invalid(grid)
    edges = np.linspace(0, RADAR, n + 1)
    ax.pcolormesh(edges, edges, masked, cmap=cmap, alpha=0.7, zorder=5)
    ax.set_title(title, color="white", pad=8)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"[OK] 已保存：{out}")


def main() -> int:
    _safe_stdout()
    parser = argparse.ArgumentParser(description="失守区域分析")
    parser.add_argument("--data", required=True, help="已解析数据目录")
    parser.add_argument("--grid", type=int, default=12, help="网格分辨率（NxN）")
    parser.add_argument("--min-samples", type=int, default=2, help="丢分率最少样本数")
    args = parser.parse_args()

    data_dir = Path(args.data)
    res = analyze(data_dir, grid=args.grid, min_samples=args.min_samples)
    records = res["records"]

    print(f"\n=== 失守区域分析 · {res['map_name']} · {res['grid']}x{res['grid']} 网格 ===")
    print("（首死数 = 回合第一个阵亡发生在此区域；丢分率 = 先掉首死后先掉一方输掉回合的比例）\n")

    # 按首死数排序展示 top 区域
    top = sorted(records, key=lambda x: (-x["opening_deaths"], -x["total_deaths"]))
    print(f"{'区域':<8}{'首死数':>6}{'丢分/首死':>10}{'丢分率':>8}{'总死亡':>8}  CT首死  T首死  世界坐标中心")
    for rec in top[:15]:
        if rec["opening_deaths"] == 0:
            continue
        zone = f"({rec['row']},{rec['col']})"
        print(
            f"{zone:<8}{rec['opening_deaths']:>6}{f'{rec['losses']}/{rec['opening_deaths']}':>10}"
            f"{rec['loss_rate']:>8.0%}{rec['total_deaths']:>8}  {rec['ct_first_deaths']:>5}  {rec['t_first_deaths']:>5}"
            f"  ({rec['world_x']},{rec['world_y']})"
        )

    # 导出 CSV
    csv_path = data_dir / "lost_zones.csv"
    pl.DataFrame(top).write_csv(csv_path)
    print(f"\n[OK] 排名表已导出：{csv_path}")

    # 输出两张图
    # 1) 首死数（危险区）
    draw_overlay(
        res["map_name"], args.grid, res["opening_deaths"], cmap="YlOrRd",
        title=f"{res['map_name']} 首死区域（danger zones）", out=data_dir / "lost_zones_opening.png",
    )
    # 2) 丢分率（需满足最少样本）
    loss_rate = {
        (r, c): (res["opening_loss"][(r, c)] / res["opening_deaths"][(r, c)])
        for (r, c), od in res["opening_deaths"].items()
        if od >= args.min_samples
    }
    draw_overlay(
        res["map_name"], args.grid, loss_rate, cmap="Reds",
        title=f"{res['map_name']} 先掉首死后丢分率（≥{args.min_samples}样本）", out=data_dir / "lost_zones_lossrate.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
