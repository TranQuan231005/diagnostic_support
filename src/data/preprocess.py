"""
VQA-Med-2019 Data Preprocessing Script
Converts raw VQA-Med-2019 QA text pairs into standard LLaVA instruction format (JSON).
Generates category-split validation files and an EDA summary report.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any


CATEGORY_FILE_MAP = {
    "C1_Modality": "Modality",
    "C2_Plane": "Plane",
    "C3_Organ": "Organ System",
    "C4_Abnormality": "Abnormality",
}


def load_category_lookup(category_dir: Path, split_suffix: str) -> Dict[Tuple[str, str], str]:
    """
    Builds a lookup map: (image_id, normalized_question) -> category_name
    from category-specific text files.
    """
    lookup = {}
    if not category_dir.exists():
        print(f"[WARN] Category directory not found: {category_dir}")
        return lookup

    for cat_prefix, cat_name in CATEGORY_FILE_MAP.items():
        cat_file = category_dir / f"{cat_prefix}_{split_suffix}.txt"
        if not cat_file.exists():
            # Try alternate naming
            possible_files = list(category_dir.glob(f"{cat_prefix}*"))
            if possible_files:
                cat_file = possible_files[0]
            else:
                print(f"[WARN] Category file not found: {cat_file}")
                continue

        with open(cat_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    img_id = parts[0].strip()
                    q = parts[1].strip().lower()
                    lookup[(img_id, q)] = cat_name

    print(f"Loaded {len(lookup)} category mappings from {category_dir}")
    return lookup


def parse_qa_file(
    qa_file: Path,
    images_dir: Path,
    category_lookup: Dict[Tuple[str, str], str],
    split_prefix: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses a pipe-delimited QA file and formats into LLaVA instruction JSON.
    Validates image existence and calculates statistics.
    """
    if not qa_file.exists():
        raise FileNotFoundError(f"QA file not found: {qa_file}")

    items = []
    missing_images = []
    stats = {
        "total_qa_pairs": 0,
        "unique_images": set(),
        "categories": Counter(),
        "answers_by_category": defaultdict(Counter),
        "question_lengths": [],
        "answer_lengths": []
    }

    with open(qa_file, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 3:
                print(f"[WARN] Malformed line {idx} in {qa_file}: '{line}'")
                continue

            img_id = parts[0].strip()
            question = parts[1].strip()
            answer = parts[2].strip()

            # Normalize image filename
            img_filename = img_id if img_id.lower().endswith((".jpg", ".png", ".jpeg")) else f"{img_id}.jpg"
            img_path = images_dir / img_filename

            if not img_path.exists():
                missing_images.append(img_filename)

            # Determine category
            cat = category_lookup.get((img_id, question.lower()), "General")
            if cat == "General":
                # Fallback: try matching on question alone
                for (lookup_img, lookup_q), lookup_cat in category_lookup.items():
                    if lookup_q == question.lower():
                        cat = lookup_cat
                        break

            item_id = f"{split_prefix}_{idx:05d}"
            llava_item = {
                "id": item_id,
                "image": img_filename,
                "category": cat,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            items.append(llava_item)

            # Update stats
            stats["total_qa_pairs"] += 1
            stats["unique_images"].add(img_filename)
            stats["categories"][cat] += 1
            stats["answers_by_category"][cat][answer.lower()] += 1
            stats["question_lengths"].append(len(question.split()))
            stats["answer_lengths"].append(len(answer.split()))

    stats["unique_images_count"] = len(stats["unique_images"])
    del stats["unique_images"]  # Not JSON serializable as set

    if missing_images:
        print(f"[WARN] {len(missing_images)} referenced images not found in {images_dir}!")
    else:
        print(f"Verified all {stats['unique_images_count']} images exist in {images_dir}")

    return items, stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess VQA-Med-2019 dataset to LLaVA JSON format.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="VQA-Med-2019",
        help="Root directory containing raw VQA-Med-2019 data folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory to save processed JSON files"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting VQA-Med-2019 Preprocessing ===")
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")

    # Training paths
    train_root = data_dir / "ImageClef-2019-VQA-Med-Training"
    train_qa_file = train_root / "All_QA_Pairs_train.txt"
    train_cat_dir = train_root / "QAPairsByCategory"
    train_img_dir = train_root / "Train_images"

    # Validation paths
    val_root = data_dir / "ImageClef-2019-VQA-Med-Validation"
    val_qa_file = val_root / "All_QA_Pairs_val.txt"
    val_cat_dir = val_root / "QAPairsByCategory"
    val_img_dir = val_root / "Val_images"

    # 1. Process Training Set
    print("\n--- Processing Training Set ---")
    train_cat_lookup = load_category_lookup(train_cat_dir, "train")
    train_items, train_stats = parse_qa_file(
        qa_file=train_qa_file,
        images_dir=train_img_dir,
        category_lookup=train_cat_lookup,
        split_prefix="train"
    )

    train_out_path = output_dir / "train.json"
    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_items)} training items to {train_out_path}")

    # 2. Process Validation Set
    print("\n--- Processing Validation Set ---")
    val_cat_lookup = load_category_lookup(val_cat_dir, "val")
    val_items, val_stats = parse_qa_file(
        qa_file=val_qa_file,
        images_dir=val_img_dir,
        category_lookup=val_cat_lookup,
        split_prefix="val"
    )

    val_out_path = output_dir / "val.json"
    with open(val_out_path, "w", encoding="utf-8") as f:
        json.dump(val_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(val_items)} validation items to {val_out_path}")

    # 3. Generate Category-Specific Validation Files
    print("\n--- Generating Category-Specific Validation Splits ---")
    cat_files = {
        "Modality": output_dir / "val_modality.json",
        "Plane": output_dir / "val_plane.json",
        "Organ System": output_dir / "val_organ.json",
        "Abnormality": output_dir / "val_abnormality.json",
    }

    for cat_name, cat_path in cat_files.items():
        cat_items = [item for item in val_items if item["category"] == cat_name]
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(cat_items, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(cat_items)} '{cat_name}' validation items to {cat_path}")

    # 4. Generate EDA Summary Report
    print("\n--- Compiling EDA Summary Report ---")
    eda_summary = {
        "train": {
            "total_qa_pairs": train_stats["total_qa_pairs"],
            "unique_images": train_stats["unique_images_count"],
            "categories": dict(train_stats["categories"]),
            "avg_question_word_length": round(sum(train_stats["question_lengths"]) / max(1, len(train_stats["question_lengths"])), 2),
            "max_question_word_length": max(train_stats["question_lengths"]) if train_stats["question_lengths"] else 0,
            "avg_answer_word_length": round(sum(train_stats["answer_lengths"]) / max(1, len(train_stats["answer_lengths"])), 2),
            "max_answer_word_length": max(train_stats["answer_lengths"]) if train_stats["answer_lengths"] else 0,
            "top_5_answers_per_category": {
                cat: dict(answers.most_common(5))
                for cat, answers in train_stats["answers_by_category"].items()
            }
        },
        "val": {
            "total_qa_pairs": val_stats["total_qa_pairs"],
            "unique_images": val_stats["unique_images_count"],
            "categories": dict(val_stats["categories"]),
            "avg_question_word_length": round(sum(val_stats["question_lengths"]) / max(1, len(val_stats["question_lengths"])), 2),
            "max_question_word_length": max(val_stats["question_lengths"]) if val_stats["question_lengths"] else 0,
            "avg_answer_word_length": round(sum(val_stats["answer_lengths"]) / max(1, len(val_stats["answer_lengths"])), 2),
            "max_answer_word_length": max(val_stats["answer_lengths"]) if val_stats["answer_lengths"] else 0,
            "top_5_answers_per_category": {
                cat: dict(answers.most_common(5))
                for cat, answers in val_stats["answers_by_category"].items()
            }
        }
    }

    eda_out_path = output_dir / "eda_summary.json"
    with open(eda_out_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved EDA summary to {eda_out_path}")

    print("\n=== Data Preprocessing Successfully Completed! ===")


if __name__ == "__main__":
    main()
