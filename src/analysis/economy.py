"""阶段 2 扩展：经济局分类（eco / force / full）。

为什么不用 awpy 的 ``round_economy.buy_type``
------------------------------------------------
awpy 只按 **全队装备总价值** 做绝对阈值切档（<5000 eco、5000~20000 force、
>=20000 full），不看两件关键事实：

1. **可用资金**——赢了手枪局第二回合人均 3500 全花光，已是经济上限，
   这叫"买满/反 eco 买"，不是强起；
2. **对手强弱**——同样一套 mp9+甲，面对对手纯手枪是优势购买，面对对手全 AK
   才是强起硬顶。

所以本模块改用三个观测量自行判定：

===========  ============================================  ==========================
观测量        含义                                          数据来源
===========  ============================================  ==========================
``spend_p``   人均本轮实际花费（意愿：买不买）                ``item_events`` 的 cost
``eq_p``      人均装备值（绝对质量）                          ``snapshots``
``n_rifle``   持长枪（步枪/狙/机枪）人数                      ``snapshots``
``eq_opp``    对手同回合 ``eq_p``（这回合值不值得打）         对手行
===========  ============================================  ==========================

判定顺序（手枪局单独成类，不参与经济分类）::

    pistol : 半场首回合（靠换边检测，见 common.pistol_round_numbers）
    eco    : spend_p < 500                  —— 没投入：攒钱 / 保枪
    full   : eq_p >= 4000 且 n_rifle >= 3   —— 绝对达标：多数人有长枪
    full   : eq_p >= 3000 且 eq_p >= 0.85*eq_opp  —— 相对不亏：对得起这回合
    force  : 其余                            —— 投入了但明显劣势：硬顶

bot 的经济行为不参与统计（bot 的 purchase cost 不可信）。

用法::

    python -m src.analysis.economy --data data/processed/<demo名>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from .common import (
    RIFLES,
    SIDE_SHORT,
    freeze_end_snapshot,
    load_map_name,
    load_table,
    pistol_round_numbers,
    safe_stdout,
)

# 可调阈值（集中在此，后续可按数据规模调整）
BUY_RULES: dict[str, float] = {
    # eco：人均实际花费低于此值视为"没在买"
    "eco_spend_max": 500,
    # full（绝对判据）：人均装备 + 长枪数同时达标
    "full_eq_min": 4000,
    "full_rifle_min": 3,
    # full（相对判据）：装备达到对手 85%，且人均装备不低于此下限
    # 下限用于避免"双方都很穷"的局被误判成 full（那种局语义上是强起）
    "full_ratio_min": 0.85,
    "full_rel_eq_min": 3000,
}

PISTOL_EQ_START_MAX = 200  # 仅供参考：手枪局全员带入装备仅剩免费手枪


def classify(
    *,
    is_pistol: bool,
    spend_p: float,
    eq_p: float,
    n_rifle: int,
    eq_opp: float,
    rules: dict[str, float] | None = None,
) -> str:
    """把一支队伍的一个回合判为 pistol / eco / force / full。"""
    r = {**BUY_RULES, **(rules or {})}
    if is_pistol:
        return "pistol"
    if spend_p < r["eco_spend_max"]:
        return "eco"
    if eq_p >= r["full_eq_min"] and n_rifle >= r["full_rifle_min"]:
        return "full"
    if eq_p >= r["full_rel_eq_min"] and eq_opp > 0 and eq_p >= r["full_ratio_min"] * eq_opp:
        return "full"
    return "force"


def eco_kind(*, spend_p: float, n_rifle: int, rules: dict[str, float] | None = None) -> str | None:
    """eco 局再细分：纯手枪省钱（eco_pure）还是带着枪省钱（eco_gear）。"""
    r = {**BUY_RULES, **(rules or {})}
    if spend_p >= r["eco_spend_max"]:
        return None
    return "eco_gear" if n_rifle >= r["full_rifle_min"] else "eco_pure"


def is_pistol_round(g: pl.DataFrame) -> bool:
    """已弃用：纯 eco 局（没人带枪）与手枪局的带入装备值无法区分。

    请用 :func:`src.analysis.common.pistol_round_numbers`（换边检测）。
    """
    raise NotImplementedError("改用 common.pistol_round_numbers()（换边检测）")


def build_round_economy(data_dir: str | Path, rules: dict[str, float] | None = None) -> pl.DataFrame:
    """构建经济分类表：每行 = 一个回合一方的经济画像 + 分类。"""
    data_dir = Path(data_dir)
    snapshots = load_table(data_dir, "snapshots_1s")  # 已剔除 bot
    rounds = load_table(data_dir, "rounds", exclude_bots=False)
    items = load_table(data_dir, "item_events")  # 已剔除 bot（bot 购买不计）

    freeze = freeze_end_snapshot(snapshots, rounds)
    pistol_rounds = set(pistol_round_numbers(snapshots, rounds))

    # 每回合每方的基础画像
    rows: list[dict] = []
    for (round_num,), grp in freeze.group_by(["round_num"], maintain_order=True):
        r = rounds.filter(pl.col("round_num") == round_num)
        if r.height == 0:
            continue
        r0 = r.row(0, named=True)
        pistol = int(round_num) in pistol_rounds

        buys = items.filter(
            (pl.col("action") == "purchase")
            & (pl.col("cost").is_not_null())
            & (pl.col("tick") >= r0["start_tick"])
            & (pl.col("tick") <= r0["freeze_end_tick"])
        )

        for side, team in grp.group_by(["side"], maintain_order=True):
            n = team.height
            if n == 0:
                continue
            spend_total = buys.filter(pl.col("side") == side[0])["cost"].sum() or 0
            rows.append(
                {
                    "round_num": int(round_num),
                    "side": side[0],
                    "n_players": n,
                    "is_pistol": pistol,
                    "spend_p": float(spend_total) / n,
                    "eq_p": float(team["equipment_value"].mean()),
                    "cash_p": float(team["money"].mean()),
                    "n_rifle": int(team["primary_weapon"].is_in(list(RIFLES)).sum()),
                }
            )

    df = pl.DataFrame(rows).with_columns(
        spend_ratio=pl.when(pl.col("spend_p") + pl.col("cash_p") > 0)
        .then(pl.col("spend_p") / (pl.col("spend_p") + pl.col("cash_p")))
        .otherwise(0.0)
        .round(3)
    )

    # 对手人均装备值
    opp = df.select(
        pl.col("round_num"),
        pl.col("side").alias("side_opp"),
        pl.col("eq_p").alias("eq_opp"),
    )
    df = (
        df.join(opp, on="round_num")
        .filter(pl.col("side") != pl.col("side_opp"))
        .with_columns(eq_ratio=(pl.col("eq_p") / pl.col("eq_opp")).round(3))
    )

    # 分类
    labels, kinds = [], []
    for row in df.iter_rows(named=True):
        labels.append(
            classify(
                is_pistol=row["is_pistol"],
                spend_p=row["spend_p"],
                eq_p=row["eq_p"],
                n_rifle=row["n_rifle"],
                eq_opp=row["eq_opp"],
                rules=rules,
            )
        )
        kinds.append(eco_kind(spend_p=row["spend_p"], n_rifle=row["n_rifle"], rules=rules))

    df = df.with_columns(
        pl.Series("buy_type_v2", labels, dtype=pl.String),
        pl.Series("eco_kind", kinds, dtype=pl.String),
    ).with_columns(saved_gear=(pl.col("eco_kind") == "eco_gear").fill_null(False))

    # 与 awpy 原分类对照
    awpy = load_table(data_dir, "round_economy", exclude_bots=False).select(
        ["round_num", "side", "buy_type"]
    )
    df = df.join(awpy, on=["round_num", "side"], how="left").rename(
        {"buy_type": "awpy_buy_type"}
    )

    cols = [
        "round_num",
        "side",
        "n_players",
        "buy_type_v2",
        "awpy_buy_type",
        "eco_kind",
        "saved_gear",
        "eq_p",
        "eq_opp",
        "eq_ratio",
        "n_rifle",
        "spend_p",
        "cash_p",
        "spend_ratio",
    ]
    return df.select(cols).sort(["round_num", "side"])


def summarize(df: pl.DataFrame) -> str:
    """生成控制台摘要文本。"""
    lines = []
    lines.append(f"回合数：{df['round_num'].n_unique()}，行数：{df.height}")

    dist = (
        df.group_by("buy_type_v2")
        .agg(pl.len().alias("行数"), pl.col("eq_p").mean().round(0).alias("人均装备"))
        .sort("buy_type_v2")
    )
    lines.append("\n== 分类分布 ==")
    for row in dist.iter_rows(named=True):
        lines.append(
            f"  {row['buy_type_v2']:<8} {row['行数']:>3} 行   人均装备 {row['人均装备']:>6}"
        )

    diff = df.filter(pl.col("buy_type_v2") != pl.col("awpy_buy_type"))
    lines.append(f"\n== 与 awpy 分类不一致：{diff.height} 行 ==")
    for row in diff.iter_rows(named=True):
        lines.append(
            f"  第 {row['round_num']:>2} 回合 {SIDE_SHORT.get(row['side'], row['side']):<2}"
            f" awpy={row['awpy_buy_type']:<6} → {row['buy_type_v2']:<6}"
            f" 人均装备 {row['eq_p']:>5.0f} / 对手 {row['eq_opp']:>5.0f}"
            f"（比例 {row['eq_ratio']:.2f}）长枪 {row['n_rifle']} 把"
            f"，花费 {row['spend_p']:>5.0f}"
        )
    return "\n".join(lines)


def main() -> int:
    safe_stdout()
    parser = argparse.ArgumentParser(description="经济局分类（eco / force / full）")
    parser.add_argument("--data", required=True, help="解析后的 demo 目录")
    parser.add_argument("--show", action="store_true", help="打印完整逐回合表格")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "tables" / "item_events.parquet").exists():
        print("❌ 缺少 tables/item_events.parquet —— 请先用新版 parse.py 重新解析该 demo")
        return 1

    df = build_round_economy(data_dir)

    out_parquet = data_dir / "tables" / "round_economy_v2.parquet"
    out_csv = data_dir / "reports" / "round_economy_v2.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_parquet)
    df.write_csv(out_csv)

    print(f"地图：{load_map_name(data_dir)}")
    print(summarize(df))
    if args.show:
        pl.Config.set_tbl_rows(100)
        pl.Config.set_tbl_width_chars(160)
        print("\n== 逐回合明细 ==")
        print(df)
    print(f"\n[完成] {out_parquet}")
    print(f"[完成] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
