using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace TransformerDemo;

/// <summary>
/// 程序入口，支持多种运行模式：
/// 1. 默认：推理示例（数字加一任务）；
/// 2. infer [目录]：加载已保存模型做推理；
/// 3. train：数字加一任务训练；
/// 4. sanguo：三国演义问答训练（从 data/sanguo_qa_*.jsonl 加载数据）。
/// </summary>
static class Program
{
    // ---------- 数字任务 ----------
    const int MaxLen = 14;
    const int BatchSize = 64;
    const int Epochs = 80;
    const float LrSgd = 0.1f;
    const float LrSpsa = 0.05f;
    const float Eps = 0.03f;
    const string ModelDir = "SavedModel";

    // ---------- 三国问答任务 ----------
    const int SanguoMaxLen = 128;
    const int SanguoBatchSize = 8;
    const int SanguoEpochs = 50;
    const float SanguoLr = 0.1f;
    const string SanguoModelDir = "SanguoModel";

    static void Main(string[] args)
    {
        args = new string[] { "ask" };
        if (args.Length > 0 && args[0] == "sanguo")
        {
            bool useGpu = args.Length > 1 && (args[1] == "gpu" || args[1] == "--gpu");
            if (useGpu)
            {
#if USE_TORCHSHARP_GPU
                TrainSanguoGpu();
#else
                Console.WriteLine("GPU 训练未启用。请先执行: dotnet build -p:UseGpu=true");
                Console.WriteLine("然后运行: dotnet run -- sanguo gpu");
                Console.WriteLine("（需 NVIDIA 显卡与 CUDA 12，如 RTX 4070）");
#endif
            }
            else
                TrainSanguo();
            return;
        }
        if (args.Length > 0 && args[0] == "generate-sanguo")
        {
            SanguoDataGeneration();
            return;
        }
        if (args.Length > 0 && args[0] == "ask")
        {
            string dir = args.Length > 1 ? args[1] : Path.Combine(AppContext.BaseDirectory, SanguoModelDir);
            SanguoInference.RunInteractive(dir);
            return;
        }
        // 数字任务
        if (args.Length > 0 && args[0] == "train")
        {
            Train();
            return;
        }
        if (args.Length > 0 && args[0] == "infer")
        {
            string dir = args.Length > 1 ? args[1] : ModelDir;
            RunInfer(dir);
            return;
        }
    }

    static void Run()
    {
        RunInfer(ModelDir);
    }

    static void RunInfer(string dir)
    {
        Console.WriteLine("--- 推理示例（加载已保存模型） ---\n");
        var examples = new[] { new[] { 1, 2, 3 }, new[] { 5, 6, 7, 8 }, new[] { 0, 9, 4 } };
        InferenceExample.Run(examples, dir);
        Console.WriteLine("按任意键退出...");
        Console.ReadKey();
    }

    /// <summary>解析三国数据路径：优先 data，其次 Data（与项目目录一致）。</summary>
    static (string trainPath, string validPath) GetSanguoDataPaths()
    {
        string baseDir = AppContext.BaseDirectory;
        foreach (string dir in new[] { "data", "Data" })
        {
            string trainPath = Path.Combine(baseDir, dir, "sanguo_qa_train.jsonl");
            string validPath = Path.Combine(baseDir, dir, "sanguo_qa_valid.jsonl");
            if (File.Exists(trainPath) || File.Exists(validPath))
                return (trainPath, validPath);
        }
        return (Path.Combine(baseDir, "data", "sanguo_qa_train.jsonl"), Path.Combine(baseDir, "data", "sanguo_qa_valid.jsonl"));
    }

#if USE_TORCHSHARP_GPU
    /// <summary>三国问答 GPU 训练（需 NVIDIA 显卡 + CUDA，如 RTX 4070）。</summary>
    static void TrainSanguoGpu()
    {
        var (trainPath, validPath) = GetSanguoDataPaths();

        var (trainData, validData, vocab) = SanguoQaDataLoader.Load(trainPath, validPath);
        if (trainData.Count == 0)
        {
            Console.WriteLine($"未找到训练数据，请将 JSONL 放入: {Path.GetFullPath(trainPath)}");
            return;
        }

        string baseDir = AppContext.BaseDirectory;
        string outDir = Path.Combine(baseDir, SanguoModelDir);
        Console.WriteLine($"三国问答 (GPU): 训练 {trainData.Count}, 验证 {validData.Count}, 词表 {vocab.Size}");
        SanguoGpuTrainer.TrainOnGpu(
            trainData, validData, vocab,
            maxLen: SanguoMaxLen,
            batchSize: SanguoBatchSize,
            epochs: SanguoEpochs,
            learningRate: SanguoLr,
            dModel: 64,
            numHeads: 2,
            dFf: 128,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            outputDir: outDir);
    }
#endif

    static void SanguoDataGeneration()
    {
        // 优先写入当前目录（项目 data），便于版本管理；否则写入运行目录下 data
        string dataDir = Path.Combine(Directory.GetCurrentDirectory(), "data");
        if (!Directory.Exists(dataDir))
            dataDir = Path.Combine(AppContext.BaseDirectory, "data");
        string trainPath = Path.Combine(dataDir, "sanguo_qa_train.jsonl");
        string validPath = Path.Combine(dataDir, "sanguo_qa_valid.jsonl");
        SanguoQaDataGenerator.GenerateToFiles(trainPath, validPath, 1000, 200);
        Console.WriteLine($"已生成: 训练集 {trainPath} (1000 条), 验证集 {validPath} (200 条)");
    }

    static void TrainSanguo()
    {
        var (trainPath, validPath) = GetSanguoDataPaths();

        var (trainData, validData, vocab) = SanguoQaDataLoader.Load(trainPath, validPath);
        if (trainData.Count == 0)
        {
            Console.WriteLine($"未找到训练数据，请将 JSONL 放入: {Path.GetFullPath(trainPath)}");
            return;
        }

        Console.WriteLine($"三国问答: 训练样本 {trainData.Count}, 验证样本 {validData.Count}, 词表大小 {vocab.Size}");

        var model = new TransformerModel(
            vocabSize: vocab.Size,
            dModel: 64,
            numHeads: 2,
            dFf: 128,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            maxLen: SanguoMaxLen);

        Console.WriteLine("开始训练 (反向传播 + SGD)...");
        for (int epoch = 1; epoch <= SanguoEpochs; epoch++)
        {
            float trainLoss = Training.TrainEpochBackpropTokenized(model, trainData, SanguoMaxLen, SanguoBatchSize, SanguoLr);
            float validLoss = Training.ValidateTokenized(model, validData, SanguoMaxLen, SanguoBatchSize);
            Console.WriteLine($"Epoch {epoch,2}: Train Loss = {trainLoss:F4}, Valid Loss = {validLoss:F4}");
        }

        string baseDir = AppContext.BaseDirectory;
        string outDir = Path.Combine(baseDir, SanguoModelDir);
        Directory.CreateDirectory(outDir);
        model.SaveToDirectory(outDir);
        vocab.SaveToDirectory(outDir);
        Console.WriteLine($"\n模型与词表已保存到: {Path.GetFullPath(outDir)}");
        Console.WriteLine("问答推理: dotnet run -- ask [模型目录]");
        Console.WriteLine("或: dotnet run -- ask " + SanguoModelDir);
    }

    static void Train()
    {
        bool useSpsa = false;

        // 构建词表与数据：任务为“输入一串 0~9，输出每位加一（模 10）”，数据在内存中随机生成
        var vocab = new Vocabulary();
        var (trainData, validData) = DataGenerator.GenerateAll();
        Console.WriteLine($"训练样本: {trainData.Count}, 验证样本: {validData.Count}, 词表大小: {vocab.Size}");

        // 小型 Transformer：8 维、1 头、16 维 FFN、各 1 层 Encoder/Decoder，便于教学与快速收敛
        var model = new TransformerModel(
            vocabSize: vocab.Size,
            dModel: 8,
            numHeads: 1,
            dFf: 16,
            numEncoderLayers: 1,
            numDecoderLayers: 1,
            maxLen: MaxLen);

        if (useSpsa)
        {
            Console.WriteLine("开始训练 (SPSA 数值梯度优化)...");
            for (int epoch = 1; epoch <= 200; epoch++)
            {
                float trainLoss = Training.TrainEpoch(model, trainData, MaxLen, BatchSize, vocab, LrSpsa, Eps);
                float validLoss = Training.Validate(model, validData, MaxLen, BatchSize, vocab);
                Console.WriteLine($"Epoch {epoch,2}: Train Loss = {trainLoss:F4}, Valid Loss = {validLoss:F4}");
            }
        }
        else
        {
            Console.WriteLine("开始训练 (反向传播 + SGD)...");
            for (int epoch = 1; epoch <= Epochs; epoch++)
            {
                float trainLoss = Training.TrainEpochBackprop(model, trainData, MaxLen, BatchSize, vocab, LrSgd);
                float validLoss = Training.Validate(model, validData, MaxLen, BatchSize, vocab);
                Console.WriteLine($"Epoch {epoch,2}: Train Loss = {trainLoss:F4}, Valid Loss = {validLoss:F4}");
            }
        }

        // 在完整验证集上计算 loss、Token 准确率、序列级准确率
        Evaluation.EvaluateOnValidation(model, validData, MaxLen, BatchSize, vocab);

        // 打印若干条“输入 -> 目标 | 预测”样例，便于直观查看模型输出
        Console.WriteLine("\n验证集样例 (输入 -> 目标 | 预测):");
        var (inputIds, targetIds, validLengths) = DataGenerator.Preprocess(
            validData.Take(3).ToList(), MaxLen, vocab);
        for (int b = 0; b < inputIds.Length; b++)
        {
            var logits = model.Forward(inputIds, targetIds, validLengths, validLengths);
            var pred = new int[targetIds[b].Length];
            for (int s = 0; s < pred.Length; s++)
            {
                int best = 0;
                for (int c = 1; c < logits[0][0].Length; c++)
                    if (logits[b][s][c] > logits[b][s][best]) best = c;
                pred[s] = best;
            }
            string inp = string.Join(",", inputIds[b].Select(id => id.ToString()));
            string tgt = string.Join(",", targetIds[b].Select(id => id.ToString()));
            string pr = string.Join(",", pred.Select(id => id.ToString()));
            Console.WriteLine($"  输入: [{inp}]");
            Console.WriteLine($"  目标: [{tgt}]");
            Console.WriteLine($"  预测: [{pr}]");
        }

        // 将模型配置与权重写入本地目录，供 infer 模式加载
        model.SaveToDirectory(ModelDir);
        Console.WriteLine($"\n模型已保存到: {System.IO.Path.GetFullPath(ModelDir)}");
        Console.WriteLine("使用已保存模型进行推理: dotnet run -- infer");
        Console.WriteLine("或指定目录: dotnet run -- infer " + ModelDir);
        Console.WriteLine("\n按任意键退出...");
        Console.ReadKey();
    }
}
