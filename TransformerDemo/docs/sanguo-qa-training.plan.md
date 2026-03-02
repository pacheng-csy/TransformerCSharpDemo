---
name: sanguo-qa-training
overview: 在现有 C# Transformer Demo 架构上，增加面向《三国演义》知识问答的训练/验证数据加载与训练流程。
todos:
  - id: design-data-format
    content: 确定三国问答数据的 JSONL 文件格式与字段约定
    status: completed
  - id: vocab-strategy
    content: 确定文本词表策略（字符级），规划 Vocabulary 扩展或重写方案
    status: completed
  - id: data-loader-design
    content: 设计从 JSONL 加载问答并转为 Sample 列表的 DataLoader 接口
    status: completed
  - id: preprocess-adaptation
    content: 规划基于新文本 Sample 的 Preprocess/batching 逻辑，保证与现有 Training 模块兼容
    status: completed
  - id: training-config
    content: 梳理针对三国问答任务的训练超参数与 Program 入口切换逻辑
    status: completed
  - id: inference-qa-api
    content: 规划训练后问答推理接口（Ask(question)）与应用对接方式
    status: in_progress
isProject: false
---

<!-- @format -->

### 目标

- **数据目标**: 基于《三国演义》构造约 1000 条高质量中短问答对，形成训练集和验证集（例如 800 / 200），用于训练一个能回答基础三国知识的 Seq2Seq 模型。
- **技术约束**: 在当前 C# Transformer Demo 的整体架构不大改的前提下，从“数字玩具任务”迁移到“文本问答任务”。
- **交付内容**: 问答数据文件格式设计、数据加载与预处理改造方案、训练流程改造方案以及如何在推理阶段使用新模型进行问答的说明。

### 一、数据格式与存储设计

- **文件格式**
  - 使用 **UTF-8 编码 JSONL 或 JSON** 存储问答，每条样本结构：
    - `{"question": "关羽在哪里被杀？", "answer": "关羽在荆州临沮被东吴吕蒙所害。"}`
  - 推荐按行存储（JSONL），便于增删与流式读取：
    - 训练集文件：`data/sanguo_qa_train.jsonl`
    - 验证集文件：`data/sanguo_qa_valid.jsonl`
  - 约定：**问题为单句/短段中文问句，答案为 1～3 句简洁回答**，避免超长上下文，便于小模型收敛。
- **内容构成建议**（你后续填充数据时的参考）
  - **人物信息**：生平、官职、重要战役，例如“诸葛亮的主要政绩有哪些？”
  - **战役事件**：官渡之战、赤壁之战、夷陵之战等过程及结果。
  - **势力格局**：三国鼎立形成过程、势力范围变化。
  - **时间线/因果关系**：某事件的前因后果，如“为什么刘备三顾茅庐？”
  - **典故梗概**：草船借箭、空城计、三英战吕布等故事简介。

### 二、文本到 token 的表示与词表方案

- **总体策略**: 保留现有 `Vocabulary` 和预处理流程的思想，但将“数字词表”替换为“文本词表”，使 `Sample.Input` / `Sample.Target` 从 `int[]` 扩展为基于文本的 token 序列。
- **词表设计**
  - **选项 A（字符级）**：
    - 将所有出现过的汉字、标点、数字、少量英文（如人名缩写）收集成字符集合，为每个字符分配一个 ID。
    - 优点：实现简单，不依赖复杂分词；缺点：序列较长、训练难度略大，但在小 Demo 中是可控的。
  - **选项 B（词/子词级）**：
    - 基于常见分词（例如按空格/自定义词典）或离线工具预生成词表；
    - 在本项目完全 C#、纯教学的前提下，建议 **先采用字符级**，后续再逐步扩展。
  - 计划采用：**字符级词表**，新增：
    - `[PAD]=0`, `[SOS]=1`, `[EOS]=2` 保持不变；
    - 后续字符从 3 开始顺序编码。
- **文本 token 化流程**（逻辑层面）
  - 遍历 JSONL 文件：对每条样本的 `question` / `answer` 分别做：
    - 去掉前后空白；
    - 按“字符”切分成 `char[]`；
    - 将每个字符通过新的 `Vocabulary`（如 `EncodeChar`）映射为 token id；
    - 构造：
      - `InputSeq = [SOS] + question_token_ids + [EOS]`
      - `TargetSeq = [SOS] + answer_token_ids + [EOS]`
  - 与当前预处理一致：后续 `Preprocess` 再负责 pad 到统一 `maxLen` 并输出 batch。

### 三、数据加载与预处理的改造思路

> 注：这里先给出“如何改造”的逻辑与关键接口，不直接改代码，便于你确认方案后再落地。

- **新增 / 改造数据结构**
  - 将 `DataGenerator.Sample` 从“整数序列”泛化为“已 token 化的 ID 序列”容器：
    - 形式保持 `record Sample(int[] Input, int[] Target)` 不变，只是数据来源改为“文本 token id 序列”。
- **新增 JSONL 读取逻辑**
  - 在类似 `DataGenerator` 的模块中增加：
    - 从文件 `sanguo_qa_train.jsonl` / `sanguo_qa_valid.jsonl` 逐行读取；
    - 用简易 JSON 解析（或者依赖 .NET 自带的 `System.Text.Json`）解析出 `question` / `answer` 字符串；
    - 调用 `Vocabulary` 的字符编码方法，得到 `int[]`；
    - 组装为 `Sample` 放入 `List<Sample>`，作为训练/验证数据返回。
- **预处理保持结构不变**
  - 延用当前 `DataGenerator.Preprocess` 的思想：
    - 输入：`IReadOnlyList<Sample> samples, int maxLen, Vocabulary vocab`
    - 输出：`(int[][] inputIds, int[][] targetIds, int[] validLengths)`
  - 区别在于：
    - 不再调用 `EncodeDigit`，而是直接使用已经 token 化好的 `Input` / `Target`；
    - 仍然在前后加 `[SOS]` / `[EOS]`，并用 `[PAD]` 补齐。

### 四、训练/验证流程如何适配新数据

- **保持训练框架不变**
  - 保持 `Training.TrainEpoch` / `TrainEpochBackprop` / `Validate` 接口和逻辑不动：
    - 它们依赖的只是 `(inputIds, targetIds, validLengths)` 和 `Vocabulary.PadId`，对“文本 vs 数字”并不敏感。
- **Program 入口处的改造思路**
  - 在 `Program.cs` 中：
    - 根据命令行模式（训练 / 推理）决定是否：
      - 加载数字玩具任务的数据（原逻辑）
      - 或加载 `sanguo_qa_train.jsonl` / `sanguo_qa_valid.jsonl` 并调用新的预处理逻辑。
  - 训练时：
    - 设定合适的 `maxLen`（例如 128 或 256，视问答长度而定）；
    - `batchSize` 根据显存/速度调节（例如 8/16）；
    - 选择使用现有的 **反向传播 + SGD** 模式做主训练路径。
- **超参数建议（针对三国问答小模型）**
  - `dModel`: 64～128 之间（视当前实现默认值而定，若已固定则沿用）；
  - `numLayers`: 2～3 层 encoder / decoder；
  - `maxLen`: 128 起步，如果你准备的问题/答案明显更长，可以提高到 256；
  - `batchSize`: 8～16；
  - `epoch`: 50～100 轮（一开始可以先跑 10～20 轮观察 loss 与样例输出）；
  - 学习率根据现有 Demo 默认值微调（保持原项目推荐值即可，必要时在训练日志中观测收敛情况）。

### 五、验证与指标设计

- **验证集划分**
  - 从总数据中预先划分：
    - 训练集约 800 条；
    - 验证集约 200 条；
    - 可以在准备数据阶段随机 shuffle 后按比例切分写入 JSONL 文件。
- **评估方式**
  - 利用已有 `Training.Validate` 计算交叉熵 loss 变化，确认模型是否收敛；
  - 额外增加**人工检查**：
    - 选取 10～20 条典型问题，将问题输入模型，查看生成答案是否事实正确、中文可读度是否可以接受。

### 六、推理与问答应用对接（高层说明）

- **推理接口复用现有 Inference 结构**
  - 按现有 `InferenceExample` 的方式：
    - 加载训练后保存的模型权重；
    - 提供一个简单函数：`string Ask(string question)`：
      - 将 `question` 用同一 `Vocabulary` token 化；
      - 构造 encoder 输入；
      - 解码时使用贪心搜索或简单 beam search 逐步生成 token，直到 `[EOS]` 或 `maxLen`；
      - 将生成的 token id 通过 `Vocabulary` 解码回汉字字符串；
      - 返回中文答案。
- **与应用层集成**
  - 后续你可以通过：
    - 控制台应用：循环读入问题、输出答案；
    - Web API：封装 `Ask` 为 HTTP 接口；
    - 桌面/前端 UI：只要调用该接口即可。

### 七、你需要准备/决定的内容

- **你来准备**
  - 填充 `sanguo_qa_train.jsonl` / `sanguo_qa_valid.jsonl` 的实际问答内容；
  - 决定问题/答案平均长度，以便最终确认 `maxLen`。
- **我后续可以帮你具体实现的**（当你从 Plan 模式切换到可执行模式后）
  - 在项目中：
    - 设计并实现字符级 `Vocabulary`（或在现有类上扩展接口）；
    - 编写 JSONL 数据加载与 token 化代码；
    - 改造 `DataGenerator` 或新增一个 `SanguoQaDataLoader`，输出与当前训练模块兼容的批数据；
    - 更新 `Program` 的训练入口，使其支持新的三国问答训练模式；
    - 编写推理函数 `Ask`，并给出调用示例。
