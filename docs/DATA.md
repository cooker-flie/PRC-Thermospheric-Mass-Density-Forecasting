# 数据和结果说明

本地包提供 Swarm-A 与 Swarm-C 的 2024 年 5 月完整筛选子集，保持原融合表字段和数值。
这是功能验证和事件演示子集，不能替代论文的多年训练数据，也不能用来验证跨年份结论。
当月含原有缺测；数据加载器保留原来的填补和窗口构建行为，未改动。
列说明见 data/column_schema.json；字段名中的单位和来源前缀沿用原项目。

数据来源入口（下载脚本可能随数据服务变化，需要访问时核实）：

- TU Delft：https://thermosphere.tudelft.nl/data/data/version_02/Swarm_data/
- OMNI：https://omniweb.gsfc.nasa.gov/
- CelesTrak：https://celestrak.org/SpaceData/
- SET：太阳代理指数的实际数据产品及再分发条款需作者确认。

完整数据处理链：下载 Swarm POD 和 OMNI、准备太阳指数原始文件，然后运行
scripts/input_data_prep/process_merge_sw_and_tudelft_data_simple.py。
使用 --help 查看 input_raw_dir、input_dir_sw_data、input_dir_sw_indices、
input_dir_omni、output_dir、satellite、date_range 参数；太阳指数原始目录的准确参数名为
--input_dir_sw_indices_raw；pymsis 用于经验密度计算。
已保留处理函数及其导入依赖，不包含重复下载历史、日志或全部原始压缩包。

results/reference 是原始年度与事件 summary.json 的只读参考快照，含逐提前量和活动分组统计。
这些汇总来自历史实验，不是本发布包新跑出的结果；权重与每个汇总的运行对应关系尚待完整追溯。
未打包数 GB 的 inference_results.csv；需要逐样本结果时用推理入口生成。
