# 经济局分类（eco / force / full）规则测试
#
# 这些用例把"为什么这么分"的战术语义固化下来，改阈值时能立刻发现回归。

import pytest

from src.analysis.economy import BUY_RULES, classify, eco_kind, is_pistol_round


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
    assert eco_kind(spend_p=300, n_rifle=4) == "eco_gear"
    assert eco_kind(spend_p=300, n_rifle=1) == "eco_pure"
    assert eco_kind(spend_p=2000, n_rifle=4) is None


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


def test_is_pistol_round_detects_empty_loadout():
    """半场首回合：全员带入装备值只剩免费手枪（<=200）。"""
    import polars as pl

    pistol = pl.DataFrame({"equipment_value_round_start": [200, 200, 200, 200, 200]})
    normal = pl.DataFrame({"equipment_value_round_start": [200, 850, 2750, 4050, 4300]})
    assert is_pistol_round(pistol) is True
    assert is_pistol_round(normal) is False
