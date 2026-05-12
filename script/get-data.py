import pandas as pd
from pathlib import Path

# 输入文件：修改为你的原始 CSV 路径
INPUT_FILE = Path("D:\protected folder\Learning\Competition\校赛\data\china_refined_oil_adjustments_2013_2026(1).csv")

# 输出文件
OUTPUT_FILE = Path("domestic_oil_adjustment_cleaned.csv")

def main():
    df = pd.read_csv(INPUT_FILE)

    # 原始字段 -> 目标字段
    rename_map = {
        "date": "日期",
        "gasoline_adjust_cny_per_ton": "汽油价格涨跌",
        "diesel_adjust_cny_per_ton": "柴油价格涨跌",
        "beijing_gasoline_ceiling_after_cny_per_ton": "汽油调整后价格",
        "beijing_diesel_ceiling_after_cny_per_ton": "柴油调整后价格",
    }

    # 检查必要字段是否存在
    missing_cols = [col for col in rename_map if col not in df.columns]
    if missing_cols:
        raise ValueError(f"原始数据缺少必要字段: {missing_cols}")

    # 只保留建模第一问需要的字段
    out = df[list(rename_map.keys())].rename(columns=rename_map)

    # 日期标准化
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    if out["日期"].isna().any():
        bad_rows = out[out["日期"].isna()]
        raise ValueError(f"存在无法解析的日期，请检查这些行:\n{bad_rows}")

    # 转换为数值，避免字符串影响后续计算
    numeric_cols = ["汽油价格涨跌", "柴油价格涨跌", "汽油调整后价格", "柴油调整后价格"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[numeric_cols].isna().any().any():
        bad_rows = out[out[numeric_cols].isna().any(axis=1)]
        raise ValueError(f"存在无法转换为数值的数据，请检查这些行:\n{bad_rows}")

    # 是否调整：
    # 汽油和柴油涨跌均为 0，则本轮未调整，通常可视为“搁浅”
    out["是否调整"] = ((out["汽油价格涨跌"] != 0) | (out["柴油价格涨跌"] != 0)).map({True: "是", False: "否"})
    out["是否搁浅"] = ((out["汽油价格涨跌"] == 0) & (out["柴油价格涨跌"] == 0)).map({True: "是", False: "否"})

    # 按日期排序
    out = out.sort_values("日期").reset_index(drop=True)

    # 日期输出为 YYYY-MM-DD
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")

    # 保存
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"处理完成，共 {len(out)} 条记录")
    print(f"输出文件：{OUTPUT_FILE.resolve()}")
    print(out.head())

if __name__ == "__main__":
    main()