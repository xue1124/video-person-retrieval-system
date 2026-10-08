# 源码整理记录

最近整理：2026-09-29。

## 范围

从原视觉项目复制必要源码到独立Git仓库。原工作目录保持不变；本目录已经连接个人GitHub仓库。

保留Vue前端、FastAPI、Celery、MySQL、FAISS、历史检测聚类、RTSP及Dify适配代码。后端按API、分析、检索、数据、任务、媒体和集成重新分组，算法行为不在此次整理中主动修改。

## 不随仓库发布的内容

- 监控录像、人物 crops、查询图片、轨迹输出、向量库、日志。
- 模型在本地 `models/` 中按PT、ONNX、TensorRT和RKNN整理，但由Git忽略，不随普通源码提交。
- 真实 `siglip.env` / `siglip.local.env`、原服务器 systemd 配置与外部包装脚本。
- 原 `schema.sql`：它包含旧表及 MySQL `sys` 导出内容，不适合作为当前新库初始化文件。
- 旧 Streamlit `app_ui.py` 和未使用的前端模板演示组件。

模型与运行资料可以留在本机用于演示，上传源码时由 `.gitignore` 排除。

## 本次实际修改

1. 增加根目录 README、模型清单、运行说明、配置模板和 Git 忽略规则。
2. 根目录 `requirements.txt` 只保留通用运行依赖；硬件相关依赖需要根据目标机器单独选择。
3. JWT 不再使用硬编码默认值；从环境文件读取私有密钥，缺少有效密钥时停止启动。
4. Celery 创建实例前先加载项目配置，避免 API 和 worker 因加载顺序不同使用不同的 Redis 设置。
5. 添加交互式账户创建工具，不提供默认密码、不覆盖已有用户。
6. 新建通用 systemd 模板，替换原机器专用路径；删除前端不存在的 favicon 引用。
7. 保留此前教学中已加入的上传抽帧步长校验。
8. 对 API 返回值和 RTSP 下载日志中的源地址做密码脱敏，避免运行凭据直接写入日志。
9. 把原先名称容易误解的分析代码归入 `services/analysis/`，增加代码地图和统一模型路径。
10. 按格式复制实际模型到本地 `models/`，并增加模型完整性检查和部署面试说明。
11. 移除旧 BoT-SORT 实验入口、离线可视化脚本、旧库回填/回滚脚本和历史环境快照；将正式流程仍使用的 OSNet 编码器拆到独立模块。
12. 将七个分阶段 SQL 和重复升级脚本合并为 `sql/schema.sql`；数据库运行时只使用 `medical_audit_v3`。
13. 统一分析工作流和入库文件命名，移除 FastAPI 的第二套启动端口与废弃检索占位函数；`run_api.py` 是唯一的本地 API 启动入口。
14. 把模型懒加载从 `tasks.py` 移到 `services/inference/`，使检索和实时流不再反向依赖 Celery 任务文件。
15. 把 Celery 配置、Worker辅助和实时截图批处理拆入 `services/tasks/`，根目录 `tasks.py` 只保留任务编排。
16. 将登录、系统状态、任务控制和报告接口拆入 `api/`；停止任务时同时更新数据库状态。
17. 移除运行时自动修改表结构的逻辑，数据库结构统一由 `sql/schema.sql` 初始化。
18. 统一运行目录和 CORS 配置，增加源码契约测试与 `.gitattributes`。

## 验证记录

- Python源码通过静态语法检查；未因此宣称GPU和外部服务已经在新结构下完成端到端验证。
- 源码契约测试检查Python语法、服务层依赖方向、数据库表清单和配置模板字段。
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
python tools/check_repository.py
git add .
git diff --cached --stat
git commit -m "Prepare video retrieval project source"
# 随后使用你自己新建仓库提供的 remote / push 命令。
```

执行commit之前仍应检查暂存文件。检查工具只检查Git候选文件，因此本地模型和运行数据可以保留，同时不会被普通 `git add .` 上传。
