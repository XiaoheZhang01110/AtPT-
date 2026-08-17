# UTA 50X 单向两阶段优化数据集

本目录由 `.github/workflows/uta50x-data-capture.yml` 自动生成数据。研究方向固定为 **West Valley Central Station → Murray Central Station**，采集从犹他当地工作日 06:30 起连续 10 个计划班次。

## 数据来源

- 线路、站点、shape 和计划时刻表：UTA 官方 GTFS static。
- 车辆位置、预测到离站时间：UTA 官方 GTFS-Realtime Vehicle Positions 与 Trip Updates。
- 高程：USGS The National Map 3DEP；沿官方 GTFS shape 等距采样 426 点。
- 信号位置、官方交叉口编号与道路名：UDOT ArcGIS `Signalized_Intersections` 图层。
- 信号相位：UDOT Automated Traffic Signal Performance Measures (ATSPM) 的实测 Timing and Actuations 图。

50X 采用感应控制及公交信号优先时，绿灯并非每天严格重复的固定窗口。因此本数据集保存与 10 班车同日、同一观测窗口的官方相位事件，并给出实测周期与绿灯时长统计；不得把统计均值冒充控制器固定配时。

## 主要输出

| 文件后缀 | 含义 |
|---|---|
| `_shape.csv` | 单向线路几何及累计距离轴 |
| `_stops.csv` | 单向站点顺序与坐标 |
| `_weekday_timetable.csv` | 该工作日全部单向计划班次 |
| `_target10_timetable.csv` | 从 06:30 起连续 10 班发车时刻表 |
| `_target10_stop_times.csv` | 10 班车逐站计划到发时刻 |
| `_elevation_426.csv` | 426 个 USGS 3DEP 高程采样点 |
| `_udot_signals.csv` | 沿线 UDOT 官方信号编号、位置、交叉口名称 |
| `_raw_vehicle_positions.csv` | 原始 GTFS-RT 车辆位置证据 |
| `_raw_trip_updates.csv` | 去重后的逐站 GTFS-RT 更新 |
| `_clean_spacetime_trajectories.csv` | 10 班清洗后的 1 s 时空轨迹和速度 |
| `_mean_speed_curve.csv` | 10 班平均速度—距离曲线 |
| `_observed_timetable.csv` | 轨迹识别的实际起讫时刻和延误 |
| `_headways.csv` | 相邻班次 9 个实测车头时距 |
| `_observed_stop_dwells.csv` | 站点停站事件与停站时间 |
| `_signal_passage_times.csv` | 各班车通过各信号的时刻 |
| `_official_green_windows_all_phases.csv` | ATSPM 全相位实测绿灯窗口 |
| `_signal_timing_status.csv` | 每个信号/相位的官方数据可得性和错误记录 |
| `_optimization_green_windows.csv` | 与公交通过时刻匹配后的线路服务相位窗口 |
| `_optimization_signal_plans.csv` | 供模型使用的相位、周期和绿灯统计 |
| `_optimization_nodes.csv` | 合并站点、信号、距离和高程的节点表 |

## 质量边界

1. `*_status.csv`、`*_summary.json` 是数据是否可用于论文的判据；缺失值不会由程序虚构。
2. ATSPM 图像提取的时间分辨率记录在 `seconds_per_pixel`，论文中应据此报告误差。
3. `phase_match_confidence` 为线路服务相位匹配置信度；低置信度信号需要人工查看同目录官方 ATSPM 原图。
4. GitHub Actions artifact 保存 90 天，应在成功后下载并存档；原始实时流无法事后补采。
