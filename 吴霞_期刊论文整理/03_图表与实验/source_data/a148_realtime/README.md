# A148实时位置数据采集说明

首尔市公交车辆位置接口只提供当前快照，不提供可回溯的完整历史轨迹。因此必须在A148工作日实际运营时连续采样。

## 认证

从韩国公共数据门户取得“서울특별시_버스위치정보조회 서비스”的服务密钥后，任选一种方式配置：

```bash
export SEOUL_BUS_API_KEY='这里填写已编码的服务密钥'
```

或者将已编码密钥单独放入本目录的`.api_key`文件。该文件已被忽略，不应上传至Overleaf或版本库。密钥不会写入代码、CSV或采集日志。

## 接口连通测试

在项目根目录执行：

```bash
python3 吴霞_期刊论文整理/03_图表与实验/redraw_scripts/collect_a148_realtime.py --once
```

## 完整往返采集

A148于工作日韩国时间03:30从上溪站发车。建议韩国时间03:20启动：

```bash
python3 吴霞_期刊论文整理/03_图表与实验/redraw_scripts/collect_a148_realtime.py \
  --interval 15 \
  --duration-minutes 220 \
  --idle-stop-minutes 15
```

15秒采样周期在220分钟内约调880次，不超过官方开发账户每日1,000次的限额。程序在车辆到达末段并消失15分钟后自动结束。

## 重建速度曲线

```bash
python3 吴霞_期刊论文整理/03_图表与实验/redraw_scripts/reconstruct_a148_speed.py \
  吴霞_期刊论文整理/03_图表与实验/source_data/a148_realtime/a148_raw_positions_YYYYMMDD.csv
```

最终CSV同时保留API时间、原始坐标、相邻位置平均速度、异常标记和重建速度。PDF中的速度是位置更新时间间的平均速度，不是车辆控制器输出的瞬时速度。

## GitHub Actions云端采集

工作流位于`.github/workflows/a148-realtime-capture.yml`，默认在工作日韩国时间03:20启动，也支持在GitHub Actions页面手动触发。

1. 将工作流、两个Python脚本、`requirements_a148.txt`以及本说明提交到仓库默认分支。
2. 进入GitHub仓库的`Settings > Secrets and variables > Actions`。
3. 新建Repository secret，名称必须为`SEOUL_BUS_API_KEY`，值为公共数据门户的已编码密钥。
4. 在`Actions > A148 real-time speed data`中手动运行，勾选`smoke_test_only`，仅核验密钥与接口连通性，不会进入220分钟采集。
5. 运行结束后，在该次workflow页面下载`a148-speed-data-*` Artifact。

云端使用Ubuntu运行器，仅生成原始数据、采集日志和重建CSV。最终Times New Roman PDF应在安装了该字体的本地Mac上生成。完成一次可用往返采集后，应在GitHub Actions中停用定时工作流，避免继续消耗付费时长。
