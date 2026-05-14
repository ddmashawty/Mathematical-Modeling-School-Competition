# 参数估计报告

## 一、数据来源与字段识别结果

- 国内调价数据: result/domastic_price.csv
- 特殊公告补充数据: data/rare-domastic.csv
- WTI 数据: data/wti-daily.csv
- Brent 数据: data/brent-daily.csv
- Basket 数据: data/basket-daily.csv
- 汇率数据: data/cny_usd_exchange_rate.csv
- 日期字段: domastic_price.date
- 公告日字段: rare-domastic.notice_date, fallback to date
- WTI 字段: wti-daily.price
- Brent 字段: brent-daily.price
- Basket 字段: basket-daily.price
- 汇率字段: cny_usd_exchange_rate second column
- 汽油价格字段: domastic_price.gasoline_price_after
- 柴油价格字段: domastic_price.diesel_price_after
- 汽油调价幅度字段: domastic_price.gasoline_change
- 柴油调价幅度字段: domastic_price.diesel_change
- 特殊调控标记字段: domastic_price.is_special_regulated

数据行数: 259。特殊调控样本数: 18。非特殊调控样本数: 241。

国际油价按公告日前最近 10 个可得交易日取均值，include_anchor=True。汇率按公告日前最近可得日汇率合并。回归时直接使用每日汇率作为已知自变量，因此模型仍为线性回归，不再额外做汇率分段常数近似；年度 segment 文件仅作为诊断输出。

## 二、汽油模型结果

- 价格损失最优 w1, w2, w3: 0.080000, 0.920000, 0.000000
- alpha: 1.219253
- beta: 3946.207005
- pi0: 665.590841
- 完整静态模型价格 MAE/RMSE/MAPE: 108.2257, 129.0764, 0.012635
- 价格波动损失最优 w1, w2, w3: 0.080000, 0.920000, 0.000000
- 价格波动 MAE/RMSE/MAPE: 108.2257, 129.0764, 0.794414
- 特殊调控 mu_i 均值/标准差/CV/有效样本数: 0.396087, 0.474197, 1.197204, 11
- mu_i 是否近似常数: not_recommended


## 三、柴油模型结果

- 价格损失最优 w1, w2, w3: 0.080000, 0.920000, 0.000000
- alpha: 1.181488
- beta: 3079.400563
- pi0: 653.706832
- 完整静态模型价格 MAE/RMSE/MAPE: 97.9008, 116.9900, 0.012910
- 价格波动损失最优 w1, w2, w3: 0.080000, 0.920000, 0.000000
- 价格波动 MAE/RMSE/MAPE: 97.9008, 116.9900, 0.743417
- 特殊调控 mu_i 均值/标准差/CV/有效样本数: 0.399007, 0.471414, 1.181469, 11
- mu_i 是否近似常数: not_recommended


## 四、汽油和柴油结果对比

- w1 差异: 0.000000
- w2 差异: 0.000000
- w3 差异: 0.000000
- alpha 差异: 0.037765
- beta 差异: 866.806442
- pi0 差异: 11.884009
- 特殊调控 mu 均值差异: 0.002920

若汽油和柴油参数存在明显差异，可能来自两类油品税费、品质差异、最高零售价基准不同，以及调价幅度四舍五入或地方口径差异。

## 五、误差较大时的可能原因

- 调价窗口可能需要改用 10 个自然日、10 个工作日或不含公告日窗口做稳健性比较。
- 国内调价日期可能存在公告日与生效日差异。
- 特殊调控标记可能仍有漏标或误标。
- 40、80、130 美元分段利润函数可能与实际政策口径存在细节差异。
- 汇率使用公告日最近值，若真实采用窗口均值，误差会扩大。
- 干净样本筛选后部分年份样本较少，年度局部参数不稳定。
- 未调价样本的实际价格包含门槛和累计影响，而静态完整模型不进行 carry 和 50 元门槛递推。
- 特殊调控样本较少，mu_i 的稳定性结论只能作为经验判断。
