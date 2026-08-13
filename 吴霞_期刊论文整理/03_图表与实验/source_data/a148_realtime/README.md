# A148公开网页实时位置采集说明

首尔公交官网只展示当前车辆位置，不提供可回溯的完整历史轨迹。因此，必须在A148实际运营期间连续保存公开网页返回的位置快照，才能重建完整速度曲线。

正式采集器使用普通无头Chromium打开`https://bus.go.kr/`，随后读取该公开页面自身使用的同源JSON资源。程序不登录、不使用API密钥、不绕过验证码，也不保存Cookie、浏览器存储、请求头或HAR文件。

## 文件与证据链

一次采集会生成以下文件：

- `a148_public_page_YYYYMMDD.png`：开始采集时的公开页面截图；
- `a148_public_route_snapshot_YYYYMMDD.json`：首次请求得到的路线元数据、线路轨迹点和41个站点；
- `a148_public_vehicle_snapshots_YYYYMMDD.jsonl`：每次请求的原始车辆数组和网页响应时间；
- `a148_raw_positions_YYYYMMDD.csv`：供速度重建器读取的标准化位置序列；
- `a148_collection_log_YYYYMMDD.jsonl`：每次请求的状态、车辆数和新增样本数；
- `a148_collection_summary_YYYYMMDD.json/.md`：采集范围和质量摘要；
- `a148_reconstructed_speed_profile_YYYYMMDD.csv`：完整性检查通过后的重建速度。

静态线路轨迹只保存一次，避免在每15秒快照中重复写入1,882个路线点。车辆数组仍按原始字段完整保留，标准化CSV只用于后续计算。

## 本地连通测试

需要Node.js和Playwright。在项目根目录执行：

```bash
node 吴霞_期刊论文整理/03_图表与实验/redraw_scripts/collect_a148_public_web.mjs \
  --once
```

`--once`只验证公开网页、A148路线和车辆位置资源是否可访问。非运营时车辆数组为0仍属于连通成功。

## 本地完整采集

A148在工作日韩国时间03:30发车。为避免GitHub托管任务延迟，云端在01:30启动；本地采集也建议在首班车前充分提前启动：

```bash
node 吴霞_期刊论文整理/03_图表与实验/redraw_scripts/collect_a148_public_web.mjs \
  --interval 15 \
  --duration-minutes 360 \
  --idle-stop-minutes 15
```

连续采集的最小允许间隔为10秒。正式配置使用15秒，最多运行6小时。程序只有在同一车辆已经覆盖起始端（区段1–2）和末端（区段40–41），且该车辆随后连续消失15分钟时才提前结束；若未观察到车辆或只得到不完整轨迹，则返回非零状态但仍保存全部证据。

## 速度重建

```bash
python 吴霞_期刊论文整理/03_图表与实验/redraw_scripts/reconstruct_a148_speed.py \
  吴霞_期刊论文整理/03_图表与实验/source_data/a148_realtime/a148_raw_positions_YYYYMMDD.csv
```

速度由相邻网页位置和时间戳计算，是采样区间平均速度，不是车辆控制器输出的瞬时速度。重建器保留长时间缺口、反向跳点和超过75 km/h位置跳变等质量标记，并默认要求去程和返程均完整。

## GitHub Actions自动采集

工作流位于`.github/workflows/a148-public-web-capture.yml`，名称为`A148 public-web speed capture`。

- 默认在每个工作日韩国时间01:30自动启动完整采集，为GitHub定时任务可能的延迟预留缓冲；
- 手动运行时默认选择`smoke_test`，只进行一次请求；
- 需要人工发起完整采集时选择`full_capture`；
- 无需创建或填写`SEOUL_BUS_API_KEY`；
- 运行结束后在该次Actions页面的`Artifacts`区域下载`a148-public-web-data-*`。

第一次运营时采集应重点核验：车辆数组字段、坐标系、网页更新时间、实际采样间隔、41个区段的覆盖范围以及往返完整性。在一次完整轨迹通过核验前，不应在论文中将结果表述为已观测的A148运营速度。

该JSON资源属于首尔公交公开网页的实现接口，不是承诺长期稳定的开放API。若页面结构或字段发生变化，采集器会停止并保留错误摘要，不会尝试规避网站限制。
