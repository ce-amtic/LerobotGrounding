from pathlib import Path

def extract_after_timestamp(
    input_path="/fyh/LerobotGrounding/finished_parquets.jsonl",
    split_marker='{"path": "2025-06-16 05:54:08"}',
    output_path="bugfix.jsonl"
):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"[错误] 输入文件不存在: {input_file}")
        return

    with input_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        marker_index = lines.index(split_marker + "\n")
    except ValueError:
        print("[警告] 未找到指定标记行，输出为空")
        marker_index = len(lines)

    # 获取之后的内容
    after_lines = lines[marker_index + 1:]

    with output_file.open("w", encoding="utf-8") as f:
        f.writelines(after_lines)

    print(f"[完成] 已保存 {len(after_lines)} 条记录到 {output_file}")

extract_after_timestamp()
