# 变更记录

## 2025-12-25 15:22

### 变更列表

- 文件路径：`docs/cn/LOCAL_DEPLOY.md`
  - 变更类型：Add
  - 函数/方法：无（文档新增）
  - 变更摘要：新增本地部署步骤与验证方式
  - 影响：用户可按步骤完成本地部署与基础验证
  - 参考：无
- 文件路径：`docs/cn/VOICE_CLONE_USAGE.md`
  - 变更类型：Add
  - 函数/方法：无（文档新增）
  - 变更摘要：新增零样本音色模拟的中文使用说明与示例代码
  - 影响：用户可使用参考音频生成相似音色
  - 参考：无
- 文件路径：`docs/cn/README.md`
  - 变更类型：Modify
  - 函数/方法：无（文档更新）
  - 变更摘要：补充本地部署与声音克隆指南链接
  - 影响：提高文档可发现性
  - 参考：无

## 2025-12-25 16:47

### 变更列表

- 文件路径：`examples/cmd/run_config.py`
  - 变更类型：Add
  - 函数/方法：main, parse_args, load_normalizer, save_audio
  - 变更摘要：新增可配置运行脚本，支持模型来源、音色、采样参数与输出格式选择
  - 影响：用户可通过命令行切换不同推理配置
  - 参考：无

- 文件路径：`run_config.bat`
  - 变更类型：Add
  - 函数/方法：无（批处理脚本）
  - 变更摘要：新增双击运行入口，支持交互式选择配置
  - 影响：用户可直接双击运行并选择参数
  - 参考：无

- 文件路径：`run_config.bat`
  - 变更类型：Modify
  - 函数/方法：无（批处理脚本）
  - 变更摘要：增加默认文本与“是否修改配置”交互分支
  - 影响：默认直接运行，必要时再手动修改配置
  - 参考：无

- 文件路径：`run_config.bat`
  - 变更类型：Modify
  - 函数/方法：无（批处理脚本）
  - 变更摘要：调整为分行注释，避免批处理中文注释误解析
  - 影响：双击与命令行运行更稳定
  - 参考：无

- 文件路径：`examples/cmd/run_config.py`
  - 变更类型：Modify
  - 函数/方法：无（模块导入路径）
  - 变更摘要：补充当前目录到模块搜索路径，保证可导入 ChatTTS
  - 影响：脚本从任意路径执行更稳定
  - 参考：无

- 文件路径：`examples/cmd/run_config.py`
  - 变更类型：Modify
  - 函数/方法：无（导入顺序）
  - 变更摘要：先加入运行路径再导入 ChatTTS，修复导入失败
  - 影响：批处理调用可正常运行
  - 参考：无

- 文件路径：`examples/cmd/run_config.py`
  - 变更类型：Modify
  - 函数/方法：parse_args
  - 变更摘要：文本参数改为可选，未输入时使用默认文案
  - 影响：可直接运行脚本而无需额外参数
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Add
  - 函数/方法：main, parse_args, load_normalizer, save_audio, read_csv_rows
  - 变更摘要：新增 CSV 批量生成与合并脚本，支持参考音频克隆与随机音色
  - 影响：可按 CSV 行批量生成并合并音频
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Add
  - 函数/方法：无（数据文件）
  - 变更摘要：新增默认 CSV 测试数据
  - 影响：可直接测试批量生成流程
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：支持“说话人”同名固定音色，参考音频可覆盖为克隆音色
  - 影响：同一角色多行保持一致音色
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：resolve_speaker_from_key, main
  - 变更摘要：新增说话人固定音色解析（file:/emb:），优先级低于参考音频
  - 影响：可用说话人字段固定音色，且支持复用
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：stable_seed_from_key, main
  - 变更摘要：为说话人生成稳定随机种子并移除分目录输出
  - 影响：同说话人音色更稳定，短音频全部输出到同一文件夹
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：parse_args, main
  - 变更摘要：参考音频改为从指定目录读取文件名（默认 input_batch）
  - 影响：CSV 只需填写文件名即可引用参考音频
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：parse_args
  - 变更摘要：默认输出目录改为 voice_output，参考音频目录改为 voice_input
  - 影响：与新目录约定一致
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.bat`
  - 变更类型：Delete
  - 函数/方法：无（批处理脚本）
  - 变更摘要：移除批处理入口，仅保留 Python 脚本
  - 影响：批量生成需使用 Python 脚本运行
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：为文本细化阶段增加稳定随机种子，提升同说话人一致性
  - 影响：同一说话人多行输出更一致
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：resolve_ref_audio_path, main
  - 变更摘要：参考音频改为“文件名前缀”，并在有参考音频时忽略说话人
  - 影响：参考音频优先级最高且无需写扩展名
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：safe_infer
  - 变更摘要：增加推理异常兜底，避免空拼接报错
  - 影响：批量运行更稳定，不因空输出中断
  - 参考：无

- 文件路径：`examples/cmd/csv_batch_utils.py`
  - 变更类型：Add
  - 函数/方法：load_normalizer, ensure_dir, save_audio, read_csv_rows, resolve_speaker_from_key, resolve_ref_audio_path, stable_seed_from_key, sanitize_text, safe_infer
  - 变更摘要：拆分批量脚本通用工具并加入文本清理与多级兜底推理
  - 影响：代码更简洁，异常时可继续生成
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：引入工具模块与文本清理，修复 GPT 直接结束导致的空输出
  - 影响：批量生成稳定性提升
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：参考音频场景不固定随机种子，降低生成空输出概率
  - 影响：参考音频克隆更稳定
  - 参考：无

- 文件路径：`examples/cmd/csv_batch_utils.py`
  - 变更类型：Modify
  - 函数/方法：save_audio, safe_infer
  - 变更摘要：延迟导入重型依赖，减少启动卡顿
  - 影响：脚本启动更快且可见输出
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：改为延迟导入并增加启动日志，定位无输出卡顿
  - 影响：运行时可见进度提示
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：参考音频改为文件名前缀示例（voice_a/voice_b）
  - 影响：示例更符合新规则
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：更新为当前参考音频前缀示例，并修复编码
  - 影响：可直接使用 voice_input 里的两条参考音频
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：缩减为单行测试数据
  - 影响：快速验证批量生成流程
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：补充说话人示例（旁白/客服A/客服B）
  - 影响：更易测试多角色对话
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.bat`
  - 变更类型：Add
  - 函数/方法：无（批处理脚本）
  - 变更摘要：新增双击运行入口，支持默认配置与可选修改
  - 影响：可直接双击运行批量生成
  - 参考：无

## 2025-12-25 21:08

### 变更列表

- 文件路径：`tools/audio/av.py`
  - 变更类型：Modify
  - 函数/方法：load_audio
  - 变更摘要：新增参考音频最大时长限制，避免异常时长导致超大内存分配与卡顿
  - 影响：参考音频读取更稳定，不易出现无响应
  - 参考：无
- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：parse_args, main
  - 变更摘要：新增参考音频最大时长参数，并在加载时应用截断
  - 影响：批量生成更稳定，参考音频克隆更可控
  - 参考：无

## 2025-12-26 09:29

### 变更列表

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：parse_args, main
  - 变更摘要：新增最小生成步数参数，避免首步 EOS 导致反复重试
  - 影响：批量生成不再卡住，提升稳定性
  - 参考：无

## 2025-12-26 09:39

### 变更列表

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：新增空音频检测与重试策略，避免生成 0 秒音频
  - 影响：生成结果更稳定，空音频会被重试或跳过
  - 参考：无

## 2025-12-26 09:44

### 变更列表

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：parse_args, main
  - 变更摘要：提高最小生成步数默认值，并增强重试最小步数
  - 影响：减少过短音频，提升成品时长稳定性
  - 参考：无

## 2025-12-26 09:51

### 变更列表

- 文件路径：`examples/cmd/csv_batch_utils.py`
  - 变更类型：Modify
  - 函数/方法：resolve_ref_audio_path
  - 变更摘要：支持带扩展名的参考音频文件名
  - 影响：参考音频匹配更稳定，减少误判为不存在
  - 参考：无

- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：main
  - 变更摘要：新增参考文本读取与传入 txt_smp，并输出参考音频使用提示
  - 影响：参考音频克隆更可用，音色更稳定
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：新增参考文本列示例，便于对齐参考音频转写
  - 影响：示例更贴近正确用法，降低误用风险
  - 参考：无

## 2025-12-26 10:06

### 变更列表

- 文件路径：`voice_input/merged_ref.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：合并两段参考音频为单一参考样本
  - 影响：便于批量示例统一使用参考音频
  - 参考：无

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：示例改用合并后的参考音频与对应转写
  - 影响：批量示例可直接引用合并样本
  - 参考：无

## 2025-12-26 10:35

### 变更列表

- 文件路径：`examples/cmd/batch_input.csv`
  - 变更类型：Modify
  - 函数/方法：无（数据文件）
  - 变更摘要：示例改用单条参考音频与对应转写
  - 影响：便于验证单条参考音频的可用性
  - 参考：无

## 2025-12-26 10:53

### 变更列表

- 文件路径：`voice_output/official_random_only.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：随机音色基线测试输出
  - 影响：用于对照判断模型是否能正常发声
  - 参考：无

- 文件路径：`voice_output/official_clone_min.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：参考音频克隆（最小步数）测试输出
  - 影响：用于验证参考音频是否导致近静音
  - 参考：无

- 文件路径：`voice_output/official_clone_trimmed.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：裁剪静音并归一化后参考音频克隆输出
  - 影响：用于验证清洗参考音频后的克隆效果
  - 参考：无

## 2025-12-26 11:25

### 变更列表

- 文件路径：`examples/cmd/csv_batch_utils.py`
  - 变更类型：Modify
  - 函数/方法：trim_and_normalize_ref
  - 变更摘要：新增参考音频静音裁剪与归一化工具，并清理重复类型导入
  - 影响：批量克隆可选清洗参考音频，降低近静音输出风险
  - 参考：无
- 文件路径：`examples/cmd/run_csv_batch.py`
  - 变更类型：Modify
  - 函数/方法：parse_args, main
  - 变更摘要：新增参考音频裁剪参数并在采样前应用
  - 影响：批量克隆更稳定，参考音频更易生效
  - 参考：无
- 文件路径：`examples/cmd/run_config.py`
  - 变更类型：Modify
  - 函数/方法：parse_args, main, trim_and_normalize_ref
  - 变更摘要：新增参考文本与静音裁剪参数，克隆场景设置最小生成步数
  - 影响：参考音频克隆更稳定，减少 0 秒输出
  - 参考：无
- 文件路径：`voice_output/official_best_run_config.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：参考音频+裁剪清洗的单次克隆输出
  - 影响：用于对比裁剪策略效果
  - 参考：无
- 文件路径：`voice_output/official_ref_txt.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：官方流程（参考音频+转写）测试输出
  - 影响：用于验证官方流程在本地的稳定性
  - 参考：无
- 文件路径：`voice_output/official_webui_style.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：模拟 WebUI 跳过细化的参考音频输出
  - 影响：用于对比跳过细化的效果
  - 参考：无
- 文件路径：`voice_output/official_merged_trim.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：合并参考音频并裁剪后的克隆输出
  - 影响：用于验证多句合并参考的可用性
  - 参考：无
- 文件路径：`voice_output/official_ref_high_sampling.mp3`
  - 变更类型：Add
  - 函数/方法：无（音频文件）
  - 变更摘要：提高采样参数后的参考音频输出
  - 影响：用于对比高采样配置的声音稳定性
  - 参考：无
- 文件路径：`voice_output/001.mp3`
  - 变更类型：Modify
  - 函数/方法：无（音频文件）
  - 变更摘要：批量生成输出覆盖为裁剪参考音频版本
  - 影响：批量示例声音更清晰
  - 参考：无
- 文件路径：`voice_output/merged.mp3`
  - 变更类型：Modify
  - 函数/方法：无（音频文件）
  - 变更摘要：合并输出覆盖为裁剪参考音频版本
  - 影响：合并结果更清晰可听
  - 参考：无

## 2025-12-26 11:35

### 变更列表

- 文件路径：`spec/PRODUCT.md`
  - 变更类型：Add
  - 函数/方法：无（文档新增）
  - 变更摘要：整理产品目标、功能与 MVP 范围
  - 影响：需求边界与目标更清晰
  - 参考：无
- 文件路径：`spec/PLAN.md`
  - 变更类型：Add
  - 函数/方法：无（文档新增）
  - 变更摘要：按阶段整理当前与后续计划
  - 影响：推进路径可追踪
  - 参考：无
- 文件路径：`spec/TECH-REFER.md`
  - 变更类型：Add
  - 函数/方法：无（文档新增）
  - 变更摘要：梳理模块关系、数据结构与调用链路
  - 影响：技术方案更易对齐
  - 参考：无
- 文件路径：`spec/TASK.md`
  - 变更类型：Add
  - 函数/方法：无（文档新增）
  - 变更摘要：记录当前进度、已完成与待办任务
  - 影响：现状搁置更可追溯
  - 参考：无
