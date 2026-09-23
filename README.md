# Tennis Vision Studio（网球视觉分析平台）

上传网球视频后，生成球员、球拍和运动网球的标注视频，并提供分段轨迹、画面速度、跟踪质量、JSON 摘要及逐帧 CSV。网页由 Gradio 提供，默认监听 `0.0.0.0:7861`。

## 实现方式

- `yolo11m.pt` 检测人员和球拍；专用 `tennis-ball.pt` 在 1280 像素输入上检测小球。
- 用球场区域、局部帧差和相邻帧位移筛选运动网球，排除部分场外人员、静置球和跳变误检。
- 对短时缺失做有距离约束的插值；在缺失、异常跳变、明显转向或约 3 秒处拆分轨迹。
- 轨迹按段翻页，已绘制的图片缓存在本次运行目录；结果保存在 `runs/YYYYMMDD/<run-id>/`。

当前跟踪是启发式位置关联，并非训练好的网球轨迹模型。速度单位为画面像素/秒；换算真实 km/h 需要摄像机及球场标定。

## 运行

使用 Python 环境安装 `requirements.txt`，并准备 `yolo11m.pt` 与网球专用权重。可通过 `TENNIS_MODEL`、`TENNIS_BALL_MODEL`、`TENNIS_DEVICE` 环境变量指定模型和设备；默认专用权重路径是 `/opt/Good-Tennis/weights/tennis-ball.pt`。然后运行：

```bash
python app.py
```

GPU 版 PyTorch 请按本机 CUDA 版本安装。`tennis-vision.service` 是当前云主机使用的 systemd 配置示例，路径和 GPU 编号需按部署环境调整。模型权重、上传视频与运行结果不包含在仓库中。
