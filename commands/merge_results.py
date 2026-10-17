import argparse
from pathlib import Path

'''
merge all repair-dataset-*.jsonl files under the input_dir into one jsonl file

python merge_results.py \
  --input_dir verusage_qwen-coder_zero-shot_VeriCoding/refinement-batch-intermediate \
  --output_file aaaaaa.jsonl
'''

def merge_jsonl_files(input_dir, output_file):
    input_path = Path(input_dir)

    jsonl_files = sorted(input_path.rglob("repair-dataset-*.jsonl"))

    if not jsonl_files:
        print("No files matched repair-dataset-*.jsonl")
        return

    total_lines = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for file in jsonl_files:
            print(f"Merging {file}")
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total_lines += 1

    print(f"\nMerged files: {len(jsonl_files)}")
    print(f"Total lines written: {total_lines}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root directory to search")
    parser.add_argument("--output_file", required=True, help="Merged jsonl output file")

    args = parser.parse_args()

    merge_jsonl_files(args.input_dir, args.output_file)