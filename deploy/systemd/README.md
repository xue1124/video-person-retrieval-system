# Linux 常驻服务模板

模板假定源码在 `/opt/osnet-siglip`、Python 在 `.venv`、系统用户为 `videoapp`。安装前按真实环境修改路径并创建对应用户，确保其可读取配置与模型、可写入运行数据目录。

模板仅覆盖 API 和 worker；MySQL、Redis、FFmpeg、GPU 依赖和前端托管需要单独准备。API 默认绑定本机，不代表已经配置 HTTPS 或公网入口。

将审阅后的模板安装到 systemd 服务目录，再通过 `systemctl enable --now` 启动。`enable` 控制开机启动，`Restart=on-failure` 控制进程异常退出后的重启。systemd 重启进程不能保证原分析任务自动恢复。

本次源码整理未安装或启动这些服务。
