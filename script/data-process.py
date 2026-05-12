"""
该文件以国内原油价格调整时间为锚点，计算前十天的平均价格
"""


import pandas as pd
from pathlib import Path


# =====================================================
# 1. 参数设置
# =====================================================

# 国内成品油价格文件
DOMESTIC_FILE = Path(r"D:\protected folder\Learning\Competition\校赛\data\domastic-price.csv")

# 国际原油日度价格文件
# 格式：date,price
CRUDE_FILE = Path(r"D:\protected folder\Learning\Competition\校赛\data\basket-daily.csv")

# 输出文件
OUTPUT_FILE = Path("basket-average-NIncl.csv")

# 是否包含国内调价当天
# False：取 [调价日前10天, 调价日前1天]
# True：取 [调价日前9天, 调价当天]
INCLUDE_ADJUST_DAY = False

# 取前多少个自然日范围
WINDOW_DAYS = 10

# 国内文件是否有表头
DOMESTIC_HAS_HEADER = False

# 国际文件是否有表头
CRUDE_HAS_HEADER = False


# =====================================================
# 2. 读取国内成品油调价数据
# =====================================================

domestic_columns = [
    "date",
    "gasoline_change",
    "diesel_change",
    "gasoline_price_after",
    "diesel_price_after",
    "is_adjusted",
    "is_shelved"
]

if DOMESTIC_HAS_HEADER:
    domestic = pd.read_csv(DOMESTIC_FILE)
else:
    domestic = pd.read_csv(
        DOMESTIC_FILE,
        header=None,
        names=domestic_columns
    )

domestic = domestic[["date"]].copy()
domestic["date"] = pd.to_datetime(domestic["date"], errors="coerce")
domestic = domestic.dropna(subset=["date"])
domestic = domestic.drop_duplicates(subset=["date"])
domestic = domestic.sort_values("date").reset_index(drop=True)


# =====================================================
# 3. 读取国际原油日度价格数据
# =====================================================

crude_columns = ["date", "price"]

if CRUDE_HAS_HEADER:
    crude = pd.read_csv(CRUDE_FILE)
else:
    crude = pd.read_csv(
        CRUDE_FILE,
        header=None,
        names=crude_columns
    )

crude = crude[["date", "price"]].copy()
crude["date"] = pd.to_datetime(crude["date"], errors="coerce")
crude["price"] = pd.to_numeric(crude["price"], errors="coerce")

crude = crude.dropna(subset=["date", "price"])
crude = crude.sort_values("date").reset_index(drop=True)


# =====================================================
# 4. 计算某个国内调价日期对应的国际油价10天范围均价
# =====================================================

def calculate_calendar_window_mean(adjust_date, crude_df):
    """
    以国内调价日期为锚点，取国际油价在前 WINDOW_DAYS 个自然日范围内的数据，
    有几条有效数据就用几条计算均价。
    """

    if INCLUDE_ADJUST_DAY:
        # 包含调价当天：
        # 例如 adjust_date = 2025-09-23
        # 窗口为 2025-09-14 到 2025-09-23，共10个自然日
        window_start = adjust_date - pd.Timedelta(days=WINDOW_DAYS - 1)
        window_end = adjust_date

        window_df = crude_df[
            (crude_df["date"] >= window_start) &
            (crude_df["date"] <= window_end)
        ]

    else:
        # 不包含调价当天：
        # 例如 adjust_date = 2025-09-23
        # 窗口为 2025-09-13 到 2025-09-22，共10个自然日
        window_start = adjust_date - pd.Timedelta(days=WINDOW_DAYS)
        window_end = adjust_date - pd.Timedelta(days=1)

        window_df = crude_df[
            (crude_df["date"] >= window_start) &
            (crude_df["date"] <= window_end)
        ]

    valid_days = len(window_df)

    if valid_days == 0:
        return pd.Series({
            "mean_price": None,
            "valid_days": 0,
            "window_start": window_start,
            "window_end": window_end
        })

    return pd.Series({
        "mean_price": window_df["price"].mean(),
        "valid_days": valid_days,
        "window_start": window_start,
        "window_end": window_end
    })


# =====================================================
# 5. 对每个国内调价日期计算窗口均价
# =====================================================

window_result = domestic["date"].apply(
    lambda x: calculate_calendar_window_mean(x, crude)
)

result = pd.concat([domestic, window_result], axis=1)


# =====================================================
# 6. 计算均价涨跌
# =====================================================

result["mean_price_change"] = result["mean_price"].diff()


# =====================================================
# 7. 整理输出
# =====================================================

result["mean_price"] = result["mean_price"].round(4)
result["mean_price_change"] = result["mean_price_change"].round(4)



# =====================================================
# 8. 同时输出详细检查版本
# =====================================================

detail_result = result.copy()

detail_result["date"] = detail_result["date"].dt.strftime("%Y-%m-%d")
detail_result["window_start"] = pd.to_datetime(
    detail_result["window_start"]
).dt.strftime("%Y-%m-%d")
detail_result["window_end"] = pd.to_datetime(
    detail_result["window_end"]
).dt.strftime("%Y-%m-%d")

DETAIL_OUTPUT_FILE = OUTPUT_FILE.with_name(
    OUTPUT_FILE.stem + "_detail.csv"
)

detail_result.to_csv(
    DETAIL_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 9. 打印信息
# =====================================================

print("处理完成")
print(f"国内数据文件: {DOMESTIC_FILE}")
print(f"国际油价文件: {CRUDE_FILE}")
print(f"窗口长度: {WINDOW_DAYS} 个自然日")
print(f"是否包含调价当天: {INCLUDE_ADJUST_DAY}")
print(f"简单输出文件: {OUTPUT_FILE}")
print(f"详细检查文件: {DETAIL_OUTPUT_FILE}")
print()
print(detail_result.head(10))