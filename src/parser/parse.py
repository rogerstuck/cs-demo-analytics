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

from awpy import Demo


def _safe_stdout() -> None:
    """让 Windows 控制台按 UTF-8 输出，避免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_demo(demo_path: str | Path, out_dir: str | Path, snapshot_seconds: float = 1.0) -> dict:
    """解析单个 demo，写出 Parquet 文件，返回汇总信息。"""
    demo_path = Path(demo_path).resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"找不到 demo 文件：{demo_path}")

    print(f"[1/3] 解析 demo：{demo_path.name}")
    demo = Demo(str(demo_path))
    demo.load("kills", "damages", "shots")

    match_id = demo_path.stem
    out = Path(out_dir) / match_id
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] 写出到：{tables_dir}")

    # 元信息
    (tables_dir / "header.json").write_text(
        json.dumps(demo.header, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 各数据集落库（Parquet 天然 UTF-8，中文名安全）
    tables: dict[str, object] = {
        "players": demo.players,
        "stats": demo.stats,
        "rounds": demo.rounds,
        "kills": demo.kills,
        "damages": demo.damages,
        "shots": demo.shots,
        "round_economy": demo.round_economy,
        "grenades": demo.grenades,
        "bomb": demo.bomb,
        "snapshots_1s": demo.snapshots(seconds=snapshot_seconds),
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
