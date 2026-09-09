# 源码整理记录

整理日期：2026-09-09。

## 范围

从原视觉项目复制必要源码到独立目录。原工作目录的视频、模型、配置和源码保留；本目录作为单独 Git 仓库，未配置远程地址、未提交、未推送。

保留 Vue 前端、FastAPI、Celery、MySQL 迁移、FAISS 检索、历史检测聚类、RTSP 及 Dify 适配代码。未大规模拆分模块或更换算法，避免整理过程中改变原业务行为。

## 不随仓库发布的内容

- 监控录像、人物 crops、查询图片、轨迹输出、向量库、日志。
- 模型权重、TensorRT/RKNN 产物、ONNX 外置权重、tokenizer 目录和 wheel 安装包。
- 真实 `siglip.env` / `siglip.local.env`、原服务器 systemd 配置与外部包装脚本。
- 原 `schema.sql`：它包含旧表及 MySQL `sys` 导出内容，不适合作为当前新库初始化文件。
- 旧 Streamlit `app_ui.py` 和未使用的前端模板演示组件。

这些资料仍保留在原项目中，本次只从上传版本排除。

## 本次实际修改

1. 增加根目录 README、模型清单、运行说明、配置模板和 Git 忽略规则。
2. 原 `requirements.txt` 保留为 `requirements/original-environment.txt`，根文件筛选直接运行依赖；硬件相关依赖单独选择。未伪造新环境验证结果。
3. JWT 不再使用硬编码默认值；从环境文件读取私有密钥，缺少有效密钥时停止启动。
4. Celery 创建实例前先加载项目配置，避免 API 和 worker 因加载顺序不同使用不同的 Redis 设置。
5. 添加交互式账户创建工具，不提供默认密码、不覆盖已有用户。
6. 新建通用 systemd 模板，替换原机器专用路径；删除前端不存在的 favicon 引用。
7. 保留此前教学中已加入的上传抽帧步长校验。
8. 对 API 返回值和 RTSP 下载日志中的源地址做密码脱敏，避免运行凭据直接写入日志。

## 验证记录

- 47 个 Python 文件通过静态语法检查；未导入完整后端启动 GPU 或连接业务库。
- `create_user.py --help` 可正常运行；账户写入未在数据库实测。
- 使用前端现有 lock 文件安装依赖，`npm run build` 通过 TypeScript 检查及 Vite 构建。
- 前端构建提示单个包超过 500 kB，这是后续拆包优化项，不是构建失败。
- Git 忽略检查确认真实 env、视频、查询图片、模型、node_modules 等不会进入普通 `git add`。
- 检查上传候选文件，未发现常见格式的真实密钥或带凭据连接串；私网 IP 仅出现于适配说明和输入占位示例。这不是全面安全审计。

未验证：全新 Python 环境依赖安装、MySQL 初始化与登录、GPU 模型推理、完整上传检索、NVR/RTSP、Dify 报告、长期稳定性、准确率对比。

## 上传方式

只上传本目录的 Git 候选文件，不能将上级工作目录整个拖入网页。依赖和构建产物留在本地，已由 `.gitignore` 排除。

建议先创建私人仓库。确认实习代码公开权限和模型/依赖许可后再决定是否公开。未自行附加 MIT/Apache 等许可证。

```bash
git status --short
python scripts/check_repository.py
git add .
git diff --cached --stat
git commit -m "Prepare video retrieval project source"
# 随后使用你自己新建仓库提供的 remote / push 命令。
```

执行 commit 之前仍可逐个检查已暂存文件。检查工具会对本地真实配置/运行素材发出提示，即使 Git 已忽略它们，目的是提醒上传目录和运行数据的区别。
