# 武器字符串规整测试
#
# 起因：demo 的 player_death 事件里，武器名可能带**涂装/名牌前缀**。
# 同一把 M4A1-S 会出现两种写法：
#   m4a1_silencer              （普通）
#   5e_2023pass1_m4a1_silencer （5E 平台 2023 赛季涂装）
# 如果不规整，按武器分组的统计会把同一把枪拆成两类。

import polars as pl

from src.parser.parse import (
    EXTRA_WEAPON_NAMES,
    canonical_weapon_names,
    normalize_weapon,
    normalize_weapon_column,
)

KNOWN = {
    "ak47",
    "m4a1",
    "m4a1_silencer",
    "awp",
    "mac10",
    "bizon",
    "glock",
    "usp_silencer",
} | set(EXTRA_WEAPON_NAMES)


def test_plain_names_unchanged():
    for w in ("ak47", "m4a1_silencer", "awp", "mac10"):
        assert normalize_weapon(w, KNOWN) == w


def test_platform_skin_prefix_is_stripped():
    """5E 平台的涂装前缀要被剥掉。"""
    assert normalize_weapon("5e_2023pass1_m4a1_silencer", KNOWN) == "m4a1_silencer"
    assert normalize_weapon("5e_2023pass1_ak47", KNOWN) == "ak47"
    assert normalize_weapon("5e_2023pass1_bizon", KNOWN) == "bizon"


def test_any_platform_prefix_works():
    """最长后缀匹配：不依赖具体平台的前缀格式。"""
    assert normalize_weapon("perfectworld_s1_ak47", KNOWN) == "ak47"
    assert normalize_weapon("faceit_2024_summer_awp", KNOWN) == "awp"
    assert normalize_weapon("some_future_platform_v9_m4a1", KNOWN) == "m4a1"


def test_weapon_prefix_is_stripped():
    """shots 表用 weapon_xxx 写法，也统一成规范名。"""
    assert normalize_weapon("weapon_ak47", KNOWN) == "ak47"
    assert normalize_weapon("weapon_m4a1_silencer", KNOWN) == "m4a1_silencer"


def test_world_and_utility_are_preserved():
    """world（非玩家武器击杀）与投掷物不能被当成前缀剥掉。"""
    assert normalize_weapon("world", KNOWN) == "world"
    assert normalize_weapon("hegrenade", KNOWN) == "hegrenade"
    assert normalize_weapon("inferno", KNOWN) == "inferno"


def test_unknown_weapon_does_not_crash():
    """未知字符串原样返回（去 weapon_ 前缀、小写），不抛异常。"""
    assert normalize_weapon("weapon_some_new_gun", KNOWN) == "some_new_gun"
    assert normalize_weapon("", KNOWN) == ""
    assert normalize_weapon(None, KNOWN) is None


def test_canonical_names_from_snapshots():
    """规范名集合从快照的主/副武器自动汇总，并补上静态武器名。"""
    snapshots = pl.DataFrame(
        {
            "primary_weapon": ["ak47", "m4a1_silencer", None, "ak47"],
            "secondary_weapon": ["glock", "usp_silencer", None, None],
        }
    )
    known = canonical_weapon_names(snapshots)
    assert {"ak47", "m4a1_silencer", "glock", "usp_silencer"} <= known
    assert {"world", "hegrenade", "knife"} <= known


def test_normalize_weapon_column_overwrites_weapon():
    """normalize_weapon_column 直接把 weapon 列改写成规范名（不新增列）。"""
    df = pl.DataFrame(
        {
            "tick": [1, 2, 3],
            "weapon": ["ak47", "5e_2023pass1_m4a1_silencer", "weapon_awp"],
        }
    )
    out = normalize_weapon_column(df, KNOWN)
    assert out.columns == ["tick", "weapon"]  # 没有多出来的列
    assert out["weapon"].to_list() == ["ak47", "m4a1_silencer", "awp"]


def test_normalize_weapon_column_skips_tables_without_weapon():
    df = pl.DataFrame({"tick": [1, 2]})
    out = normalize_weapon_column(df, KNOWN)
    assert out.columns == ["tick"]
