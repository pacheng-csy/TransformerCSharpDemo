## 项目简介

`transformer-demo` 是一个**纯 C# 实现的小型 Transformer 教学项目**，主要用于演示：

- 如何从零实现一个简化版 Transformer（包含多头注意力、LayerNorm、前馈网络等）；
- 如何将模型应用到两类任务：
  - **数字玩具任务**：输入一串 0~9，输出“每位加一”的序列；
  - **《三国演义》问答任务**：基于 JSONL 问答数据，训练字符级 Seq2Seq 模型；
- 如何在同一套模型/训练框架下，切换不同的数据与任务，并可选接入 **TorchSharp + CUDA** 做 GPU 训练。

所有核心逻辑（矩阵运算、前向与反向传播、训练循环、推理逻辑）均使用 C# 自行实现，适合作为学习 Transformer 原理与 C# 数值计算的参考代码。

---

## 环境要求

- **运行环境**
  - .NET SDK **9.0**（`TargetFramework: net9.0`）
  - Windows（示例和 GPU 包版本以 Windows + CUDA 12 为主）

- **可选：GPU 训练（TorchSharp）**
  - 需要 NVIDIA 显卡（例如 RTX 4070）、支持 CUDA 12；
  - 编译时显式启用：
    - `dotnet build -p:UseGpu=true`
  - `TransformerDemo.csproj` 中会自动引用：
    - `TorchSharp`
    - `libtorch-cuda-12.1-win-x64`

---

## 目录结构概览

项目主要结构（简化版）如下：

```text
transformer-demo/
├── TransformerDemo/                 # C# 控制台项目（核心代码）
│   ├── Program.cs                   # 程序入口，选择任务/模式
│   ├── TransformerDemo.csproj       # 项目文件（含 GPU 条件引用）
│   ├── 项目结构说明.md              # 项目内部结构与阅读指引
│   ├── data/                        # 三国问答 JSONL 数据（运行时会复制到输出目录）
│   ├── Data/                        # 词表与数据相关代码（数字 + 三国问答）
│   ├── Model/                       # Transformer 模型实现
│   ├── Training/                    # 训练与评估逻辑
│   ├── Inference/                   # 推理逻辑（数字示例 + 三国问答）
│   └── docs/                        # 架构 / 流程 / 数据结构等教学文档
└── README.md                        # 本文件
```

更详细的项目结构说明请阅读：

- `TransformerDemo/项目结构说明.md`
- `TransformerDemo/docs/架构总览.md`

---

## 快速开始

以下命令均假定当前目录为仓库根目录 `transformer-demo`。

### 1. 恢复依赖并编译

```bash
dotnet restore
dotnet build
```

如果需要启用 GPU（TorchSharp + CUDA），请使用：

```bash
dotnet build -p:UseGpu=true
```

### 2. 数字玩具任务

#### 2.1 训练（每位加一）

```bash
dotnet run --project TransformerDemo -- train
```

训练过程中会在控制台输出每个 epoch 的训练/验证 loss，并在结束时进行一次验证集评估。

#### 2.2 推理示例

训练完成后，模型权重会保存到 `TransformerDemo/SavedModel`（或运行目录下的同名文件夹）。你可以通过：

```bash
dotnet run --project TransformerDemo -- infer
```

来加载已保存模型，并对若干整数序列做“每位加一”的推理示例。

---

## 三国演义问答任务

### 1. 数据准备

三国问答数据存放在：

- `TransformerDemo/data/sanguo_qa_train.jsonl`
- `TransformerDemo/data/sanguo_qa_valid.jsonl`

数据格式说明详见：

- `TransformerDemo/Data/README.md`

如果你还没有准备自己的数据，可以使用项目内置的数据生成器自动生成一份示例数据：

```bash
dotnet run --project TransformerDemo -- generate-sanguo
```

该命令会在 `data/` 目录下生成（或覆盖）`sanguo_qa_train.jsonl` 与 `sanguo_qa_valid.jsonl` 文件。

### 2. CPU 训练

使用 CPU 训练三国问答模型：

```bash
dotnet run --project TransformerDemo -- sanguo
```

流程简要：

- 读取 JSONL 问答数据；
- 构建字符级词表 `CharVocabulary`；
- 将问答转为 `DataGenerator.Sample` 列表；
- 通过 `Training.TrainEpochBackpropTokenized` 进行若干 epoch 训练；
- 在验证集上计算 loss；
- 最终将模型权重与 `vocab.json` 保存到 `SanguoModel` 目录。

### 3. GPU 训练（可选）

在具备 CUDA 的 NVIDIA 显卡上，可以使用 TorchSharp 在 GPU 上训练更大的模型：

```bash
dotnet build -p:UseGpu=true
dotnet run --project TransformerDemo -- sanguo gpu
```

GPU 训练由 `SanguoGpuTrainer` 负责，训练结束后同样会将权重导出为 C# 模型可直接使用的格式，并与字符词表一起保存到指定目录。

### 4. 交互式问答推理

当你已经有一个训练好的三国问答模型目录（例如 `SanguoModel`）后，可以使用：

```bash
dotnet run --project TransformerDemo -- ask
```

或显式指定模型目录：

```bash
dotnet run --project TransformerDemo -- ask SanguoModel
```

进入交互式问答模式后：

- 在控制台输入任意关于《三国演义》的问题；
- 程序会调用 `SanguoInference.Ask` 对问题进行编码并逐步生成答案；
- 输入空行或 `exit` 即可退出。

---

## 学习建议与文档导航

如果你希望**系统性学习整个项目**，可以按照以下顺序阅读：

1. **项目鸟瞰**
   - `TransformerDemo/项目结构说明.md`
   - `TransformerDemo/docs/架构总览.md`
2. **数据与预处理**
   - 数字任务：
     - `TransformerDemo/Data/Vocabulary.cs`
     - `TransformerDemo/Data/DataGenerator.cs`
   - 三国问答：
     - `TransformerDemo/Data/CharVocabulary.cs`
     - `TransformerDemo/Data/SanguoQaDataLoader.cs`
     - `TransformerDemo/Data/SanguoQaDataGenerator.cs`
     - `TransformerDemo/Data/README.md`（数据格式）
3. **核心数学与模型结构**
   - `TransformerDemo/Core/MatrixHelper.cs`
   - `TransformerDemo/Model/` 下各文件（Embedding、Attention、Encoder/Decoder、TransformerModel 等）
4. **训练与推理**
   - `TransformerDemo/Training/Training.cs`
   - `TransformerDemo/Training/Evaluation.cs`
   - `TransformerDemo/Inference/InferenceExample.cs`
   - `TransformerDemo/Inference/SanguoInference.cs`
5. **补充教学文档**
   - `TransformerDemo/docs/训练与推理流程.md`
   - `TransformerDemo/docs/数据结构与核心类型.md`

阅读顺序建议从简单到复杂、从“用法”到“实现细节”，在实际跑通训练/推理之后再回到源码中深入理解每一步的数学含义与实现方式。

