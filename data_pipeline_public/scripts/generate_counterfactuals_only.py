"""
从已有的 dialogues 文件生成反事实样本。
不重新跑对话，只跑 Phase 4。

用法:
    cd data_pipeline
    python scripts/generate_counterfactuals_only.py --dialogues data/dialogues_5turn_shard0000.jsonl
    # 或批量:
    python scripts/generate_counterfactuals_only.py --dialogues data/dialogues_5turn_shard*.jsonl
"""

import argparse
import asyncio
import json
import glob
from pathlib import Path

from utils_io import read_jsonl, write_jsonl
from _llm import CFG

# 复用主脚本的反事实生成函数
from importlib import import_module


async def main():
    parser = argparse.ArgumentParser(description="Generate counterfactuals from existing dialogues")
    parser.add_argument("--dialogues", nargs="+", required=True,
                        help="Dialogue jsonl files (supports glob)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Concurrent sessions for counterfactual generation")
    args = parser.parse_args()

    # 展开 glob
    dialogue_files = []
    for pattern in args.dialogues:
        dialogue_files.extend(glob.glob(pattern))
    dialogue_files = sorted(set(dialogue_files))

    if not dialogue_files:
        print("[error] No dialogue files found")
        return

    data_dir = Path(__file__).resolve().parents[1] / "data"

    # 加载 profiles 和 ads
    profiles = list(read_jsonl(data_dir / "profiles_.jsonl"))
    profile_map = {p["profile_id"]: p for p in profiles}
    all_ads = list(read_jsonl(data_dir / "ads_corpus_synthetic.jsonl"))
    print(f"Loaded {len(profiles)} profiles, {len(all_ads)} ads")

    # 导入反事实生成函数
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from importlib.machinery import SourceFileLoader
    gen_module = SourceFileLoader("gen", str(Path(__file__).parent / "03_generate_pure_dialogues.py")).load_module()
    generate_counterfactuals_for_session = gen_module.generate_counterfactuals_for_session

    for df_path in dialogue_files:
        print(f"\n== Processing {df_path} ==")
        sessions = list(read_jsonl(df_path))
        print(f"  {len(sessions)} sessions")

        # 输出文件: dialogues_xxx.jsonl → counterfactuals_xxx.jsonl
        df_name = Path(df_path).name
        cf_name = df_name.replace("dialogues_", "counterfactuals_")
        cf_path = Path(df_path).parent / cf_name
        print(f"  Output: {cf_path}")

        all_cf = []
        batch_size = args.batch_size
        for i in range(0, len(sessions), batch_size):
            batch = sessions[i:i + batch_size]
            coros = [
                generate_counterfactuals_for_session(sess, profile_map[sess["profile_id"]], all_ads)
                for sess in batch if sess["profile_id"] in profile_map
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    print(f"  [error] counterfactual: {r}")
                else:
                    all_cf.extend(r)
            done = min(i + batch_size, len(sessions))
            print(f"  counterfactual progress: {done}/{len(sessions)} sessions ({len(all_cf)} samples)")

        write_jsonl(cf_path, all_cf)
        n_cf_ad = sum(1 for r in all_cf if r["sample_type"] == "counterfactual_ad")
        n_cf_early = sum(1 for r in all_cf if r["sample_type"] == "target_early")
        print(f"  Wrote {len(all_cf)} counterfactual samples (ad: {n_cf_ad}, early: {n_cf_early})")


if __name__ == "__main__":
    asyncio.run(main())
