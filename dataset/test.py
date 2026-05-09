import json
import os

def process_json_data(input_file, output_file):
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    try:
        # 1. 读取原始 JSON 数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 批量删除指定字段
        # 假设数据是一个列表（List），如果是单个对象则直接操作
        if isinstance(data, list):
            for item in data:
                remove_keys(item)
        elif isinstance(data, dict):
            remove_keys(data)

        # 3. 将处理后的数据写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文/特殊字符不被转码，indent=4 保持格式美观
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"处理完成！结果已保存至: {output_file}")

    except Exception as e:
        print(f"处理过程中出现异常: {e}")

def remove_keys(obj):
    """删除单个字典对象中的指定字段"""
    keys_to_remove = ["validation_status"]
    for key in keys_to_remove:
        if key in obj:
            del obj[key]

if __name__ == "__main__":
    # 配置文件名
    input_filename = "dataset/testset_general_check_test.json"
    output_filename = "dataset/testset_general_check_test.json"

    process_json_data(input_filename, output_filename)