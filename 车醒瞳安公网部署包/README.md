# 车醒瞳安

基于眼部可见性、眼睛状态、头部姿态与时间序列分析的疲劳驾驶风险识别智能体（纯软件原型）。

## 当前阶段

第一阶段不接硬件、不连接车辆，只处理视频分析模块输出的结构化特征。这样可以先验证风险判断逻辑，并避免在早期把问题混在摄像头、芯片和车辆接口上。

当前原型可以：

- 判断眼部是否可见，并对墨镜/遮挡场景降级处理；
- 根据 EAR（眼睛纵横比）、闭眼持续时间、打哈欠次数和头部姿态计算风险；
- 对单帧结果做时间序列平滑，避免一次眨眼触发疲劳报警；
- 输出风险等级、可信度、原因和建议动作。

## 运行演示

在项目根目录运行：

```powershell
python car_ai_agent\src\run_demo.py
```

下一步再接入 MediaPipe/OpenCV，把视频帧转成 `FrameFeatures`，核心判断模块不需要重写。

## 第二阶段特征流测试

当前环境如果没有安装 OpenCV/MediaPipe，可以先运行结构化特征流：

```powershell
python car_ai_agent\src\run_csv_demo.py
```

CSV 的字段定义见 `data/sample_features.csv`。未来的视觉模型只需要按同样字段输出，即可接入核心智能体。

安装视频依赖后，`src/video_adapter.py` 可以把本地视频逐帧转换为同样的 `FrameFeatures`。当前适配器先实现眼睛闭合和嘴部张开特征；瞳孔追踪、头部姿态和墨镜可见性分类会在后续迭代中加入。

## 第三阶段 本地视频分析

准备一段驾驶员正面视频后，运行：

```powershell
python car_ai_agent\src\analyze_video.py C:\视频路径\sample.mp4 --out car_ai_agent\reports\sample_report.json
```

程序只在本地读取视频，当前不会调用摄像头、上传视频或控制车辆。视频中没有检测到人脸时，会按“眼部不可见、证据不足”处理，而不会强行判断为疲劳。

## 第四阶段 比赛演示视频

放入一段视频后，可以生成带状态面板的演示视频和报告：

```powershell
python car_ai_agent\src\render_analysis_video.py car_ai_agent\data\raw\sample.mp4 `
  --out-video car_ai_agent\reports\sample_annotated.mp4 `
  --out-report car_ai_agent\reports\sample_report.json
```

建议先阅读 `data/raw/README.md`，按四种典型状态采集本地测试视频。

当前第一轮样本校准采用 `EAR < 0.20` 判断闭眼、`MAR > 0.45` 判断疑似打哈欠。阈值来自项目测试样本，仅适用于原型验证；后续应增加不同人员、光照和拍摄角度的数据后再重新标定。

哈欠不会由单帧张嘴直接确认：嘴部需超过个人阈值并持续约 1 至 8 秒，且随后恢复闭合后，系统才记录一次哈欠。该规则用于降低正常说话、短暂表情造成的误判。

## 本地测试网页与智能复盘

运行下面命令后，在浏览器打开 `http://127.0.0.1:5000`：

```powershell
python car_ai_agent\web_app.py
```

网页支持上传视频、生成标注视频并下载 JSON 报告。实时检测停止后，页面会生成不含视频的结构化训练摘要，并提供本地“智能复盘 Agent”：教练可以询问本次训练的风险原因、疲劳与监测中断区别、持续低头情况和是否建议休息。复盘回答只基于本次摘要，不调用外部模型、不上传视频。

实时模式支持“个人校准”：启动摄像头后先自然注视镜头约 10 秒，系统会以个人正常 EAR、MAR 和头部姿态为基线，再设置闭眼、打哈欠和姿态偏离阈值。校准只保存在当前内存会话中，停止检测后清除。

产品定位不是替代车规级 DMS，而是面向驾校、汽车实训基地和中小企业的本地化驾驶员状态分析与训练复盘工具。系统将结果分为疲劳事件、分心/姿态异常和监测中断，并在证据不足时明确告知用户。

## 第五阶段 测试数据与指标

`data/evaluation_template.csv` 用于建立事件级测试集；每一行代表一个可复核片段，需填写人工标签、系统输出、可见性和响应时间。运行下面命令会生成 `metrics.json` 和场景准确率图：

```powershell
python car_ai_agent\src\evaluate_cases.py car_ai_agent\data\evaluation_template.csv --out-dir car_ai_agent\reports\evaluation
```

`data/evaluation_demo.csv` 和 `reports/evaluation_demo` 仅演示评估流程，不能当作正式准确率。正式比赛材料应替换为经两名成员复核的真实标注数据，并注明样本数量、参与者数量和场景构成。

## 目录

- `src/fatigue_agent.py`：风险判断核心；
- `src/review_agent.py`：基于结构化检测摘要的本地复盘 Agent；
- `src/run_demo.py`：无视频、无硬件的离线演示；
- `config/risk_policy.json`：阈值和策略配置；
- `data/raw`：后续放原始视频；
- `reports`：后续放分析报告。

本项目输出仅用于研究和比赛原型，不作为真实车辆安全系统或医疗判断依据。
