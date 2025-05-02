import os
import json

data_dir = "wohuman"  # 根目录
output_path = "wohuman_instructions.json"

instruction_dict = {}

for root, dirs, files in os.walk(data_dir):
    for fname in files:
        if fname.endswith(".json"):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                episode_id = os.path.relpath(fpath, data_dir).replace("/", "__").replace(".json", "")
                instruction = data.get("instructions", "").strip()
                if instruction:
                    instruction_dict[episode_id] = instruction
            except Exception as e:
                print(f"[ERROR] Failed to read {fpath}: {e}")

# 保存到 instructions.json
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(instruction_dict, f, indent=2, ensure_ascii=False)

print(f"[DONE] Extracted {len(instruction_dict)} instructions to {output_path}")