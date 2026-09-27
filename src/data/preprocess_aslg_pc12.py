from pathlib import Path
from itertools import zip_longest
import csv
import gzip
import io
import re
import string
import unicodedata
import zipfile


def normalize_text(text, make_uppercase=False):
    """Normalize text without removing meaningful ASL notation."""

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Replace repeated whitespace with one space
    text = re.sub(r"\s+", " ", text).strip()

    if make_uppercase:
        return text.upper()

    return text.lower()


# Original dataset folder
DATASET_DIR = Path(
    r"C:\Users\HP\OneDrive\Documents\NeuraSign_Datasets"
)

# Processed data will remain outside the GitHub repository
OUTPUT_DIR = (
    DATASET_DIR
    / "processed"
    / "aslg_pc12_official"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all official ASLG-PC12 corpus ZIP files
zip_files = sorted(
    DATASET_DIR.glob("corpus_*.clean.zip")
)

print("=" * 60)
print("ASLG-PC12 DATASET PREPROCESSING")
print("=" * 60)

print(f"Dataset folder: {DATASET_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Corpus ZIP files found: {len(zip_files)}")

print("\nCorpus files:")

for zip_path in zip_files:
    print(f" - {zip_path.name}")

# Confirm that all 16 official files are available
if len(zip_files) != 16:
    raise FileNotFoundError(
        f"Expected 16 corpus ZIP files, "
        f"but found {len(zip_files)}."
    )

print("\nAll 16 official corpus files are available.")


# ---------------------------------------------------------
# Step 1: Validate English and ASL line alignment
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CHECKING ENGLISH-ASL LINE ALIGNMENT")
print("=" * 60)

total_english_lines = 0
total_asl_lines = 0
mismatched_files = []

for zip_path in zip_files:
    corpus_name = zip_path.stem

    english_filename = f"{corpus_name}.en.txt"
    asl_filename = f"{corpus_name}.asl.txt"

    english_count = 0
    asl_count = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive_files = archive.namelist()

        if english_filename not in archive_files:
            raise FileNotFoundError(
                f"{english_filename} is missing "
                f"from {zip_path.name}."
            )

        if asl_filename not in archive_files:
            raise FileNotFoundError(
                f"{asl_filename} is missing "
                f"from {zip_path.name}."
            )

        with archive.open(english_filename) as english_raw:
            with archive.open(asl_filename) as asl_raw:

                english_file = io.TextIOWrapper(
                    english_raw,
                    encoding="utf-8",
                    errors="replace"
                )

                asl_file = io.TextIOWrapper(
                    asl_raw,
                    encoding="utf-8",
                    errors="replace"
                )

                for english_line, asl_line in zip_longest(
                    english_file,
                    asl_file
                ):
                    if english_line is not None:
                        english_count += 1

                    if asl_line is not None:
                        asl_count += 1

    total_english_lines += english_count
    total_asl_lines += asl_count

    if english_count == asl_count:
        status = "MATCHED"
    else:
        status = "MISMATCHED"
        mismatched_files.append(zip_path.name)

    print(
        f"{zip_path.name}: "
        f"English={english_count:,}, "
        f"ASL={asl_count:,}, "
        f"Status={status}"
    )

print("\nDataset validation summary")
print(f"Total English lines: {total_english_lines:,}")
print(f"Total ASL lines: {total_asl_lines:,}")
print(
    f"Mismatched corpus files: "
    f"{len(mismatched_files)}"
)


# ---------------------------------------------------------
# Step 2: Preprocess every record from all 16 ZIP files
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING COMPLETE PREPROCESSING")
print("=" * 60)

total_processed_rows = 0
total_valid_pairs = 0
total_invalid_pairs = 0

summary_rows = []

for zip_path in zip_files:
    corpus_name = zip_path.stem

    english_filename = f"{corpus_name}.en.txt"
    asl_filename = f"{corpus_name}.asl.txt"

    output_file = (
        OUTPUT_DIR
        / f"{corpus_name}.processed.csv.gz"
    )

    fieldnames = [
        "source_corpus",
        "line_number",
        "english_clean",
        "asl_gloss_clean",
        "english_word_count",
        "asl_gloss_count",
        "english_character_count",
        "punctuation_count",
        "has_question_mark",
        "valid_pair",
    ]

    processed_rows = 0
    valid_pairs = 0
    invalid_pairs = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(english_filename) as english_raw:
            with archive.open(asl_filename) as asl_raw:

                english_file = io.TextIOWrapper(
                    english_raw,
                    encoding="utf-8",
                    errors="replace"
                )

                asl_file = io.TextIOWrapper(
                    asl_raw,
                    encoding="utf-8",
                    errors="replace"
                )

                with gzip.open(
                    output_file,
                    mode="wt",
                    encoding="utf-8",
                    newline=""
                ) as csv_file:

                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=fieldnames
                    )

                    writer.writeheader()

                    for line_number, pair in enumerate(
                        zip_longest(
                            english_file,
                            asl_file
                        ),
                        start=1
                    ):
                        english_line, asl_line = pair

                        english_clean = normalize_text(
                            english_line or ""
                        )

                        asl_clean = normalize_text(
                            asl_line or "",
                            make_uppercase=True
                        )

                        valid_pair = bool(
                            english_clean and asl_clean
                        )

                        if valid_pair:
                            valid_pairs += 1
                        else:
                            invalid_pairs += 1

                        english_word_count = len(
                            english_clean.split()
                        )

                        asl_gloss_count = len(
                            asl_clean.split()
                        )

                        punctuation_count = sum(
                            character in string.punctuation
                            for character in english_clean
                        )

                        writer.writerow({
                            "source_corpus": corpus_name,
                            "line_number": line_number,
                            "english_clean": english_clean,
                            "asl_gloss_clean": asl_clean,
                            "english_word_count":
                                english_word_count,
                            "asl_gloss_count":
                                asl_gloss_count,
                            "english_character_count":
                                len(english_clean),
                            "punctuation_count":
                                punctuation_count,
                            "has_question_mark":
                                int("?" in english_clean),
                            "valid_pair":
                                int(valid_pair),
                        })

                        processed_rows += 1

    total_processed_rows += processed_rows
    total_valid_pairs += valid_pairs
    total_invalid_pairs += invalid_pairs

    summary_rows.append({
        "source_corpus": corpus_name,
        "processed_rows": processed_rows,
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "output_file": output_file.name,
    })

    print(
        f"Completed {zip_path.name}: "
        f"{processed_rows:,} rows, "
        f"{valid_pairs:,} valid pairs, "
        f"{invalid_pairs:,} invalid pairs"
    )


# ---------------------------------------------------------
# Step 3: Save preprocessing summary
# ---------------------------------------------------------

summary_file = (
    OUTPUT_DIR
    / "aslg_pc12_preprocessing_summary.csv"
)

with summary_file.open(
    mode="w",
    encoding="utf-8",
    newline=""
) as file:

    summary_writer = csv.DictWriter(
        file,
        fieldnames=[
            "source_corpus",
            "processed_rows",
            "valid_pairs",
            "invalid_pairs",
            "output_file",
        ]
    )

    summary_writer.writeheader()
    summary_writer.writerows(summary_rows)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(
    f"Total processed rows: "
    f"{total_processed_rows:,}"
)

print(
    f"Total valid English-ASL pairs: "
    f"{total_valid_pairs:,}"
)

print(
    f"Total invalid pairs: "
    f"{total_invalid_pairs:,}"
)

print(
    f"Processed files saved in: "
    f"{OUTPUT_DIR}"
)

print(
    f"Summary file: "
    f"{summary_file}"
)