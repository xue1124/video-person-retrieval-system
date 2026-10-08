# 本地运行说明

先运行本地上传和图片检索；NVR、实时流、Dify 报告分别依赖额外服务。

## 1. 环境

建议使用独立 Python 环境。源码使用 Python 3.10+ 语法，原部署使用过 Python 3.12。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 改用 .\.venv\Scripts\Activate.ps1
```

先安装匹配机器驱动的 PyTorch/torchvision，以及 `onnxruntime-gpu` 或 `onnxruntime` 中的一种，再安装：

```bash
python -m pip install -r requirements.txt
```

此依赖列表从原环境筛选，未进行全新环境依赖解析。不要在工作环境中盲目升级；遇到依赖冲突需在独立环境校验。TensorRT/RKNN 为可选硬件路线，不在基础安装中强制安装。

安装 MySQL 8、Redis 和 FFmpeg，使 `ffmpeg`、`ffprobe` 可通过 PATH 找到。所有 Python 启动命令都在仓库根目录执行，因为部分模型读取使用相对路径。

## 2. 私有配置

复制 `deploy/env/siglip.env.example` 为同目录的 `siglip.env`，填写数据库连接和 JWT 密钥。此真实配置已被 `.gitignore` 排除。

生成密钥：

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

将输出作为 `JWT_SECRET`，不要提交或分享。配置解析器读取普通 `KEY=VALUE`，不要在值外面添加引号或在值后写行尾注释。数据库密码包含特殊字符时需要 URL 编码。

数据库名固定为 `medical_audit_v3`。名称中的 `v3` 只是最终结构版本，不代表运行时还连接 v2；请使用单独的开发 MySQL 实例或确认该名称没有存放已有业务数据。

## 3. 初始化空数据库

在 MySQL 客户端依次执行下面文件（路径按仓库实际位置填写）：

在 DBeaver 中连接 MySQL，打开并完整执行 `sql/schema.sql`。它会创建正式系统所需的 17 张表；不需要再依次执行补丁文件。

如需检查已有本地数据库与检索索引：

```bash
python tools/verify_database.py
```

创建首个账户：

```bash
python tools/create_user.py --username admin --role admin
```

按提示输入密码，脚本保存 bcrypt 哈希，不生成默认密码，也不覆盖已有用户。

## 4. 准备模型和启动

按 [模型清单](MODELS.md) 补齐兼容模型、外置 ONNX 权重和 tokenizer。API 的查询路径与 worker 的建模路径必须使用同一版编码器。

在不同终端从仓库根目录运行：

```bash
python run_api.py
python -m celery -A tasks.app worker --loglevel=info --pool=threads --concurrency=1
```

然后：

```bash
cd frontend
npm ci
npm run dev
```

`npm run dev` 用于本地开发；提交前可额外运行一次 `npm run build`，检查 TypeScript 并生成生产构建产物。二者不是必须同时常驻运行。

若前端 Node 版本不满足依赖要求，需要切换到符合 lock 文件 engines 的版本。Linux 常驻服务模板在 `deploy/systemd/`，需按本机路径和用户修改。不要将 Windows `solo` 池当作多任务并发。

## 5. 验证一条最小链路

1. 请求 `http://127.0.0.1:8002/health`，确认 API 可响应（不代表所有模型可用）。
2. 打开 `http://localhost:5174` 并登录新建账户。
3. 上传有权使用的短视频，确认任务从等待进入分析并完成。
4. 用该视频里的人物截图进行 OSNet 检索，确认能返回图片、时间并回放。
5. 再测试文字查询及删除功能。测试素材和查询图片都留在本地。

不依赖数据库和模型的源码契约测试：

```bash
python -m unittest discover -s tests -v
```

## 外部服务

`services/integrations/nvr_tunnel.py` 依赖 ISAPI HTTP 代理、RTSP 中继控制接口及流服务器；本仓库没有这些服务的实现。仅修改 NVR 地址不能保证接入可运行。

Dify 报告需另外配置工作流 URL、密钥和符合 `services/reports/dify_client.py` 输入输出约定的工作流。本仓库不自动创建工作流。
