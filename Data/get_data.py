import akshare as ak
import pandas as pd


import csv
GLD = []
# with open('VOT.csv', 'r') as file:
#     reader = csv.reader(file)
#     a=next(reader) # 如果有标题行，取消这行注释来跳过它
#     a.append("return")
#     GLD.append(a)
#     for row in reader:
#         Ratereturn=(float(row[4])-float(row[1]))/float(row[1])
#         row.append(Ratereturn)
#         GLD.append(row)
#
#
# df = pd.DataFrame(GLD)
# df.to_csv("VOT.csv", index=False, encoding='utf-8')

import pandas as pd

import pandas as pd

# 1. 读取两个 CSV 文件
try:
    df_users = pd.read_csv('GLD.csv')
    df_orders = pd.read_csv('VOT.csv')

    print("用户信息:\n", df_users)
    print("\n订单信息:\n", df_orders)

    # 2. 水平合并 (merge)
    # on='UserID' 告诉 pandas 使用 'UserID' 这一列作为匹配的“键”
    # 默认 'how=inner'，只保留两个文件**都能**匹配上的行
    merged_df = pd.merge(df_users, df_orders, on='date', how='left')

    # 3. 保存到新文件
    merged_df.to_csv('VOT_GLD.csv', index=False, encoding='utf-8')

    print(f"\n--- 合并成功! ---")
    print("结果已保存到 'VOT_GLD.csv':\n", merged_df)

except FileNotFoundError:
    print("错误: 找不到文件。请确保 'users.csv' 和 'orders.csv' 存在。")
except KeyError:
    print("错误: 找不到共同的键 'UserID'。请检查您的 CSV 标题行。")
except Exception as e:
    print(f"发生错误: {e}")