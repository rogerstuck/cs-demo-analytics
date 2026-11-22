# 经济局分类（eco / force / full）规则测试
#
# 这些用例把"为什么这么分"的战术语义固化下来，改阈值时能立刻发现回归。

import pytest

from src.analysis.common import pistol_round_numbers
from src.analysis.economy import BUY_RULES, classify, eco_kind


def _cls(spend_p, eq_p, n_rifle, eq_opp, is_pistol=False):
    return classify(
        is_pistol=is_pistol, spend_p=spend_p, eq_p=eq_p, n_rifle=n_rifle, eq_opp=eq_opp
    )


def test_pistol_round_is_its_own_class():
    """手枪局单独成类，不参与经济分类（即使花光 800）。"""
    assert _cls(800, 800, 0, 800, is_pistol=True) == "pistol"


def test_eco_when_almost_no_money_spent():
    """几乎不花钱 = eco（攒钱 / 保枪）。"""
    assert _cls(spend_p=0, eq_p=900, n_rifle=0, eq_opp=4000) == "eco"
    assert _cls(spend_p=BUY_RULES["eco_spend_max"] - 1, eq_p=900, n_rifle=1, eq_opp=4000) == "eco"


def test_eco_gear_distinguished_from_pure_eco():
    """保枪 eco（带长枪不花钱）与纯手枪 eco 要能分开。"""
    assert eco_kind(buy_type="eco", spend_p=300, n_rifle=4) == "eco_gear"
    assert eco_kind(buy_type="eco", spend_p=300, n_rifle=1) == "eco_pure"
    assert eco_kind(buy_type="eco", spend_p=2000, n_rifle=4) is None


def test_eco_kind_only_applies_to_eco():
    """eco_kind 只对 buy_type == "eco" 有意义。

    手枪局人均花费同样是 0，若不加限制会被归成 eco_pure，
    按 eco_kind 分组统计时手枪局就会混进纯 eco。
    """
    assert eco_kind(buy_type="pistol", spend_p=0, n_rifle=0) is None
    assert eco_kind(buy_type="force", spend_p=3000, n_rifle=1) is None
    assert eco_kind(buy_type="full", spend_p=2000, n_rifle=5) is None


def test_full_by_absolute_standard():
    """人均 >= 4000 且多数人有长枪 = 全买。"""
    assert _cls(spend_p=3000, eq_p=4800, n_rifle=5, eq_opp=4800) == "full"


def test_round2_after_pistol_win_is_full_not_force():
    """赢下手枪局第二回合：人均 3500 已是经济上限（对手只有手枪）→ 全买，不是强起。

    对应 Sonkyo_2026-11-16_de_dust2_1 第 2 回合 CT：eq_p=3500，对手 1370。
    """
    assert _cls(spend_p=2250, eq_p=3500, n_rifle=2, eq_opp=1370) == "full"


def test_force_when_outgunned_despite_spending_everything():
    """同样一套微冲+甲，但对手是全 AK → 强起。

    对应第 17 回合 CT：eq_p≈3930 / 对手 5060，花费比例 96%。
    """
    assert _cls(spend_p=3220, eq_p=3930, n_rifle=2, eq_opp=5060) == "force"


def test_both_teams_broke_is_force_not_full():
    """双方都穷时，相对占优也不算 full（FULL_REL_EQ_MIN 下限的作用）。

    对应第 14 回合：T 人均 2850 / CT 1880，两边都没买到正经装备。
    """
    assert _cls(spend_p=2350, eq_p=2850, n_rifle=1, eq_opp=1880) == "force"


def test_relative_rule_rescues_low_absolute_but_adequate_buy():
    """装备不到 4000 但打到对手 85% 以上，且自身不是破枪 → full。"""
    assert _cls(spend_p=3300, eq_p=3850, n_rifle=4, eq_opp=3900) == "full"


def test_zero_opponent_equipment_does_not_divide_by_zero():
    """对手装备为 0（异常数据）时不应误判 full。"""
    assert _cls(spend_p=2000, eq_p=2500, n_rifle=1, eq_opp=0) == "force"


@pytest.mark.parametrize("custom", [{"full_rel_eq_min": 2000}])
def test_rules_are_overridable(custom):
    """阈值可通过入参覆盖（双穷局是否算 full 是个可调取舍）。"""
    got = classify(
        is_pistol=False,
        spend_p=2350,
        eq_p=2850,
        n_rifle=1,
        eq_opp=1880,
        rules=custom,
    )
    assert got == "full"


def test_pistol_rounds_detected_by_side_flip_not_by_loadout():
    """手枪局靠**换边**判定，不靠"带入装备只剩免费手枪"。

    纯 eco 局（没人带枪回来）带入装备与手枪局一模一样，必须能区分。
    """
    import polars as pl

    rounds = pl.DataFrame(
        {
            "round_num": [1, 2, 3, 4],
            "start_tick": [100, 200, 300, 400],
            "end_tick": [190, 290, 390, 490],
        }
    )
    rows = []
    for rn, start in [(1, 100), (2, 200), (3, 300), (4, 400)]:
        for sid in (11, 12):
            # 第 1~2 回合 sid 11/12 在 CT；第 3 回合起换边到 T
            side = "counter-terrorist" if rn <= 2 else "terrorist"
            rows.append({"tick": start + 10, "steamid": sid, "side": side})
        for sid in (21, 22):
            side = "terrorist" if rn <= 2 else "counter-terrorist"
            rows.append({"tick": start + 10, "steamid": sid, "side": side})
    snapshots = pl.DataFrame(rows)

    # 第 1 回合（首回合）与第 3 回合（换边）是手枪局，其他不是
    assert pistol_round_numbers(snapshots, rounds) == [1, 3]
