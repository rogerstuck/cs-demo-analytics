# 经济分类端到端测试：用合成的表格目录验证 build_round_economy 的完整链路
#
# 单元测试（test_economy.py）只覆盖 classify() 纯函数；这里造一份最小的
# tables/ 目录（snapshots_1s / rounds / item_events / round_economy），
# 真正跑一遍读表 → freeze 末快照 → 购买汇总 → 分类 → 输出。
#
# 合成回合设计（以 CT 方视角；R6 为换边后的手枪局，双方阵营互换）：
#   R1 手枪局      全员只有免费手枪（eq_start=200）            → pistol
#   R2 保枪局      上回合保下 AK（eq_start=4700），只花 150/人  → eco + eco_gear
#   R3 纯 eco      只有手枪，一分不花，人均剩 2000             → eco + eco_pure
#   R4 强起       人均 3500、1 把长枪、对手 5000，花光         → force
#   R5 全买       人均 5000、5 把长枪                         → full
#   R6 手枪局      下半场首回合（换边），全员手枪            → pistol
#
# 另含 bot（steamid=0）与观战者数据，验证它们被自动剔除。

import json

import polars as pl
import pytest

from src.analysis.economy import build_round_economy

STEAMIDS = [76561198000000001 + i for i in range(5)]  # CT 5 名真人
T_STEAMIDS = [76561198000000011 + i for i in range(5)]  # T 5 名真人
BOT_STEAMID = 0

# (回合号, CT 人均装备, CT 保枪带入值, CT 长枪数, CT 人均花费, CT 人均剩余)
CT_PROFILE = [
    (1, 800, 200, 0, 0.0, 0),  # 手枪局
    (2, 5000, 4700, 5, 150.0, 1500),  # 保枪局：带枪不花钱
    (3, 200, 200, 0, 0.0, 2000),  # 纯 eco
    (4, 3500, 200, 1, 3000.0, 200),  # 强起：花光但装备不够
    (5, 5000, 200, 5, 2000.0, 50),  # 全买
    (6, 800, 200, 0, 0.0, 0),  # 下半场手枪局
]
T_PROFILE = [  # T 方全程正常全买，作为对照基准（人均 5000）
    (1, 800, 200, 0, 0.0, 0),
    (2, 5000, 200, 5, 3000.0, 500),
    (3, 5000, 200, 5, 3000.0, 500),
    (4, 5000, 200, 5, 3000.0, 500),
    (5, 5000, 200, 5, 3000.0, 500),
    (6, 800, 200, 0, 0.0, 0),
]


def _rosters(idx: int):
    """返回本回合 (CT 的 steamid 列表, T 的 steamid 列表)；R6 起换边。"""
    if idx < 5:
        return STEAMIDS, T_STEAMIDS
    return T_STEAMIDS, STEAMIDS  # 下半场换边

RIFLE = "ak47"
PISTOL = "glock"


def _make_tables_dir(tmp_path) -> "object":
    """构造一份最小可用的 tables/ 目录（列名与 parse.py 产出保持一致）。"""
    data_dir = tmp_path / "fake_demo"
    tables = data_dir / "tables"
    tables.mkdir(parents=True)

    (tables / "header.json").write_text(
        json.dumps({"map_name": "de_dust2"}, ensure_ascii=False), encoding="utf-8"
    )

    rounds, snapshot_rows, purchase_rows, eco_rows = [], [], [], []

    for idx, (rn, *_) in enumerate(CT_PROFILE):
        start = 1000 * (idx + 1)
        freeze_end = start + 500
        rounds.append(
            {
                "round_num": rn,
                "start_tick": start,
                "freeze_end_tick": freeze_end,
                "end_tick": start + 900,
                "winner_side": "counter-terrorist",
            }
        )
        tick = freeze_end - 10  # 买入已完成、尚未开局的快照

        ct_ids, t_ids = _rosters(idx)
        for profile, steamids, side in (
            (CT_PROFILE[idx], ct_ids, "counter-terrorist"),
            (T_PROFILE[idx], t_ids, "terrorist"),
        ):
            _, eq, eq_start, n_rifle, spend_p, cash = profile
            for i, sid in enumerate(steamids):
                snapshot_rows.append(
                    {
                        "tick": tick,
                        "steamid": sid,
                        "name": f"{side[:2]}{i}",
                        "side": side,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "primary_weapon": RIFLE if i < n_rifle else PISTOL,
                        "equipment_value": eq,
                        "equipment_value_round_start": eq_start,
                        "money": cash,
                    }
                )
            # 该方本轮购买事件：人均花费均摊到每人一笔
            if spend_p > 0:
                for sid in steamids:
                    purchase_rows.append(
                        {
                            "tick": start + 100,
                            "steamid": sid,
                            "name": "buyer",
                            "side": side,
                            "action": "purchase",
                            "item": RIFLE,
                            "cost": int(spend_p),
                        }
                    )
            eco_rows.append(
                {
                    "round_num": rn,
                    "side": side,
                    "equipment_value": eq * 5,
                    "buy_type": "placeholder",
                    "n_players": 5,
                }
            )

    # bot 与观战者：必须被剔除，不能影响任何统计
    snapshot_rows.append(
        {
            "tick": 1010,
            "steamid": BOT_STEAMID,
            "name": "Darryl",
            "side": "counter-terrorist",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "primary_weapon": RIFLE,
            "equipment_value": 99999,  # 极端值：若未被剔除会显著拉高 eq_p
            "equipment_value_round_start": 99999,
            "money": 99999,
        }
    )
    purchase_rows.append(
        {
            "tick": 1100,
            "steamid": BOT_STEAMID,
            "name": "Darryl",
            "side": "counter-terrorist",
            "action": "purchase",
            "item": RIFLE,
            "cost": 8000,  # bot 的 cost 不可信，同样不能计入
        }
    )

    pl.DataFrame(rounds).write_parquet(tables / "rounds.parquet")
    pl.DataFrame(snapshot_rows).write_parquet(tables / "snapshots_1s.parquet")
    pl.DataFrame(purchase_rows).write_parquet(tables / "item_events.parquet")
    pl.DataFrame(eco_rows).write_parquet(tables / "round_economy.parquet")

    return data_dir


@pytest.fixture(scope="module")
def economy_df(tmp_path_factory):
    data_dir = _make_tables_dir(tmp_path_factory.mktemp("synthetic"))
    return build_round_economy(data_dir)


def _ct(df, rn):
    return df.filter(
        (pl.col("round_num") == rn) & (pl.col("side") == "counter-terrorist")
    ).row(0, named=True)


def test_pistol_round(economy_df):
    row = _ct(economy_df, 1)
    assert row["buy_type_v2"] == "pistol"
    assert row["n_rifle"] == 0


def test_gear_save_round_is_eco_with_saved_gear_flag(economy_df):
    """保枪局：全员带 AK、人均只花 150 → eco，且 saved_gear=True。"""
    row = _ct(economy_df, 2)
    assert row["spend_p"] == pytest.approx(150.0)
    assert row["eq_p"] == pytest.approx(5000.0)
    assert row["n_rifle"] == 5
    assert row["buy_type_v2"] == "eco"
    assert row["eco_kind"] == "eco_gear"
    assert row["saved_gear"] is True


def test_pure_eco_round(economy_df):
    """纯 eco：只有手枪、一分不花 → eco + eco_pure。"""
    row = _ct(economy_df, 3)
    assert row["buy_type_v2"] == "eco"
    assert row["eco_kind"] == "eco_pure"
    assert row["saved_gear"] is False


def test_force_round(economy_df):
    """强起：人均 3500、1 把长枪、对手 5000、钱花光 → force。"""
    row = _ct(economy_df, 4)
    assert row["buy_type_v2"] == "force"
    assert row["n_rifle"] == 1
    assert row["eq_ratio"] == pytest.approx(0.7)
    assert row["spend_ratio"] > 0.9


def test_full_round(economy_df):
    """全买：人均 5000、5 把长枪 → full。"""
    row = _ct(economy_df, 5)
    assert row["buy_type_v2"] == "full"


def test_second_half_pistol_round(economy_df):
    """下半场换边后的首回合也应该是手枪局。"""
    assert _ct(economy_df, 6)["buy_type_v2"] == "pistol"


def test_pure_eco_is_not_mistaken_for_pistol(economy_df):
    """回归测试：纯 eco 局（没人带枪，带入装备值同样是 200）不能被当成手枪局。"""
    row = _ct(economy_df, 3)
    assert row["buy_type_v2"] != "pistol"
    assert row["buy_type_v2"] == "eco"


def test_bot_rows_are_excluded(economy_df):
    """bot（steamid=0）的装备与购买花费不能进入统计。"""
    for rn in (1, 2, 3, 4, 5, 6):
        row = _ct(economy_df, rn)
        assert row["n_players"] == 5, "bot 被计入人数"
        assert row["eq_p"] < 99999, "bot 的装备值污染了 eq_p"
    # R2：若 bot 的 8000 花费被计入，人均花费会远高于 150
    assert _ct(economy_df, 2)["spend_p"] == pytest.approx(150.0)


def test_every_round_has_both_sides(economy_df):
    """每回合两方都要有记录（对手列不能串行）。"""
    counts = economy_df.group_by("round_num").agg(pl.len().alias("n")).sort("round_num")
    assert counts["n"].to_list() == [2, 2, 2, 2, 2, 2]
