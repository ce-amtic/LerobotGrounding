"""
python run_inspection.py
"""

import os
import json
import subprocess

# --- 配置 ---
# 存放JSON文件的目录名
INSPECTION_DIR = "inspection_images"

# 需要运行的脚本名
TARGET_SCRIPT = "lerobot_explorer.py"

# --- 脚本主逻辑 ---

def main():
    """
    主函数，执行所有操作。
    """
    # 1. 检查目标目录是否存在
    if not os.path.isdir(INSPECTION_DIR):
        print(f"错误：目录 '{INSPECTION_DIR}' 不存在。")
        print("请在脚本所在位置创建该目录，并将JSON文件放入其中。")
        return

    # 2. 获取目录下所有的json文件名，并进行排序以保证处理顺序
    try:
        json_files = sorted([f for f in os.listdir(INSPECTION_DIR) if f.endswith(".json")])
    except FileNotFoundError:
        print(f"错误：在列出文件时，目录 '{INSPECTION_DIR}' 找不到了。")
        return

    if not json_files:
        print(f"在 '{INSPECTION_DIR}' 目录中没有找到 .json 文件。")
        return

    print(f"找到 {len(json_files)} 个JSON文件。开始逐个处理...\n")

    # 3. 逐个处理JSON文件
    for i, filename in enumerate(json_files):
        json_path = os.path.join(INSPECTION_DIR, filename)
        
        print("-" * 60)
        print(f"({i+1}/{len(json_files)}) 正在处理文件: {json_path}")

        try:
            # 4. 读取并解析JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 5. 提取并显示关键字信息
            record_idx = data.get("record_idx", "未找到")
            task_desc = data.get("task_desc", "未找到")
            parquet_path = data.get("parquet_path")
            
            # 6. 检查parquet_path是否存在，如果不存在则无法运行后续脚本
            if parquet_path is None:
                print("\n警告：在此文件中未找到 'parquet_path' 关键字，无法运行目标脚本。")
            else:
                print(f"  - parquet_path: {parquet_path}")
                print("\n即将运行以下命令:")
                command_str = f'python {TARGET_SCRIPT} --parquet-file "{parquet_path}"'
                print(f"  {command_str}\n")
                
                # 7. 构建并运行另一个Python脚本
                try:
                    # 使用列表形式传递命令更安全
                    command_list = ["python", TARGET_SCRIPT, "--parquet-file", parquet_path]
                    subprocess.run(command_list, check=True)
                    print(f"\n脚本 '{TARGET_SCRIPT}' 执行完毕。")

                except FileNotFoundError:
                    print(f"错误：无法找到 '{TARGET_SCRIPT}' 或 'python' 命令。")
                    print("请确保 Python 已安装并在系统 PATH 中，并且该脚本与本脚本在同一目录下。")
                    break  # 中断整个流程
                except subprocess.CalledProcessError as e:
                    print(f"错误：运行 '{TARGET_SCRIPT}' 时出错，返回码: {e.returncode}")
                    # 你可以选择在这里中断或继续
                    # break

        except json.JSONDecodeError:
            print(f"错误：无法解析 {filename}。文件可能不是有效的JSON格式。")
        except Exception as e:
            print(f"处理 {filename} 时发生未知错误: {e}")

        # 8. 等待用户确认，除非是最后一个文件

        print(f"  - record_idx: {record_idx}")
        print(f"  - task_desc:  {task_desc}")
        print(f"  - parquet_path: {parquet_path}")

        if i < len(json_files) - 1:
            try:
                input("\n>>> 按 Enter 键继续处理下一个文件，或按 Ctrl+C 退出...")
            except KeyboardInterrupt:
                print("\n\n用户中断操作。正在退出。")
                break
        
    print("-" * 60)
    print("所有文件处理完毕。")


if __name__ == "__main__":
    main()