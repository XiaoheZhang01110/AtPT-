# HSL 20单向运营数据采集说明

本目录对应赫尔辛基HSL 20路的单向运营数据，固定采集方向为：

`Eira → Munkkivuori`（`route_id=1020`，`direction_id=0`）。

A148继续保留为前期探索案例；HSL 20路用于采集更具常规城市公交代表性的纯电动公交运行数据。

## 官方数据源

- 静态GTFS：`https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip`
- GTFS-Realtime车辆位置：`https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl`

两项数据均不需要API密钥。静态GTFS用于确定站点、站序、线路形状和计划时刻；实时车辆位置用于采集经纬度、时间戳、车辆编号、下一站、运行状态、方向、方位角和HSL报告速度。

截至2026-08-14的GTFS核验结果为：单向约11.029 km，共25个站点，计划单程约33–44分钟，具体运行时间随班次而变化。

## 沿线高程与信号节点

用于线路—高程图的补充数据于2026-08-15整理：

- 高程源：Open-Meteo Elevation API，对官方GTFS线形的257个点逐点采样；
- 高程范围：0–28 m；
- 信号节点源：OpenStreetMap `highway=traffic_signals`；
- 匹配方法：保留距GTFS线形50 m以内的OSM节点，并将沿线里程相距75 m以内的相邻信号节点聚类；
- 聚类后得到33处沿线信号位置。

33处OSM位置已进一步与赫尔辛基官方WFS图层`Liikennevalot_piste`匹配：

- 33/33处均获得官方交叉口编号和名称；
- 对应31个不同的官方交叉口，因为两组相邻OSM位置分别属于同一官方交叉口；
- 31处为高置信匹配（距离不超过35 m），1处为中置信，1处需要人工复核；
- 官方WFS提供交叉口编号、名称、类型、位置和更新时间，不提供周期或相位配时。

高程属于公开DEM派生数据，不是实地测量值。OSM中一个交叉口可能包含多个信号灯节点，因此论文图中使用聚类后的运营位置，不将其解释为信号灯硬件数量。

## 已完成的实测单程

2026-08-14已完成一条可审计的单向轨迹：

- 运行标识：`20260814_171000_22-1257`；
- 车辆编号：`22/1257`；
- 计划发车：17:10（赫尔辛基当地时间）；
- 实际观测区间：17:08:56–17:57:26；
- 轨迹观测时长：48.5分钟（包含车辆在起点发车前的停留）；
- 有效时间戳：475个；
- 站点覆盖：25/25；
- 线路投影覆盖：0.021–10.978 km（官方线路总长11.029 km）；
- 车辆报告速度：均值13.53 km/h，最大值40.32 km/h；
- 位置反算速度：有效区间均值13.08 km/h，最大值37.88 km/h；
- 465/475个样本通过全部区间质量检查，另有3个长缺口、3个位置速度异常和4个反向位置跳点被保留并标记；
- 地图匹配最大误差：15.03 m。

原始单向采集还包含同一时段其他方向0车辆，用于证据保全；`_speed_profile.csv`只保留上述完整单程。

## 生成文件

文件均使用`hsl20_eira_to_munkkivuori_YYYYMMDD`前缀：

- `_metadata.json`：线路、方向、首末站、里程和当日班次数量；
- `_stops.csv`：25个站点的站序、坐标和线路里程；
- `_timetable.csv`：采集日全部单向计划班次；
- `_shape.csv`：单向GTFS线路形状；
- `_first_feed.pb`：第一次请求保存的原始GTFS-Realtime证据；
- `_filtered_snapshots.jsonl`：每次请求中筛选出的20路方向0车辆；
- `_raw_positions.csv`：标准化实时位置数据；
- `_collection_log.jsonl`：逐次请求日志；
- `_collection_summary.json/.md`：完整性和覆盖范围摘要；
- `_speed_profile.csv`：完整单程的地图匹配与速度重建结果；
- `_speed_profile_summary.json`：速度曲线质量摘要。
- `_elevation.csv`：257个GTFS线形点对应的公开DEM高程；
- `_osm_signals.csv`：路线走廊内经匹配和聚类的OSM信号节点；
- `_route_context_metadata.json`：高程与信号节点的数据源、数量和匹配参数。
- `_helsinki_official_signals.geojson`：匹配日下载的赫尔辛基官方交通信号WFS快照；
- `_signals_official_match.csv`：33处OSM位置到官方编号和交叉口名称的逐点匹配表；
- `_official_signal_intersections.csv`：按线路里程排序的31个去重官方交叉口；
- `_signals_official_match_summary.json`：匹配数量、距离和置信等级摘要。

## 完整单程判定

采集器同时记录方向0的全部车辆，但只有同一`车辆编号 + 运营日期 + 计划发车时间`满足以下条件才判为完整单程：

1. 在Eira首站或其100 m范围内被观测；
2. 此后明确观测到Munkkivuori末站ID，并继续跟踪至车辆进入末站75 m范围；
3. 两端观测相隔至少15分钟；
4. 至少获得60个不同时间戳样本。

未达到完整性要求时保留原始证据，但重建器默认拒绝输出“完整速度曲线”，避免把中途截取的车辆轨迹误当成完整单程。

20路在进入Munkkivuori环线前会提前经过距离终点较近的位置，而且HSL会在车辆仍驶向终点时提前把下一站切换为终点ID，并可能在实际到达时清空下一站字段。因此末端不能只按距离或只按单帧站点ID判定；采集器必须先看到终点站ID `1304161`，随后继续观察到车辆进入终点75 m范围才结束。

## 两种速度

- `speed_reported_kmh`：HSL在GTFS-Realtime中直接提供的车辆报告速度；
- `speed_position_kmh`：相邻GPS点投影到GTFS线路形状后，根据线路里程差和时间差计算的区间平均速度。

论文绘图建议以车辆报告速度为主曲线，以位置重建速度作为质量核验。长时间缺口、位置反跳、超过75 km/h的异常速度以及超过120 m的地图匹配误差均会单独标记，不进行跨缺口插值。

## GitHub Actions

工作流为`.github/workflows/hsl20-single-direction-capture.yml`，显示名称为`HSL 20 single-direction speed capture`。

- 手动运行默认选择`smoke_test`，只请求一次；
- 选择`full_capture`后，最多连续采集120分钟，目标是获得3条完整单程；
- 采集器保留同一时段的全部方向0车辆，达到3条完整运行后提前结束；
- 重建器同时输出`_speed_profiles.csv`和`_speed_profiles_summary.json`，每条记录包含`profile_index`和`run_id`；
- 工作日自动任务在UTC 04:30启动，即芬兰冬季06:30、夏季07:30；
- 完成后在该次Actions运行页面的`Artifacts`区域下载`hsl20-single-direction-data-*`；
- 不需要创建任何GitHub Secret。
