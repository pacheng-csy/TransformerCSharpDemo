using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TransformerDemo;

/// <summary>
/// 程序入口，支持三种运行模式：
/// 1. 推理模式（infer）：加载已保存的模型权重，仅做前向推理演示，不训练；
/// 2. 训练模式（默认）：使用反向传播 + SGD 训练一个小型 Transformer，保存模型并打印预测样例；
/// 3. 训练模式（spsa）：使用 SPSA 数值梯度优化训练，便于对比理解“无反向传播”时的收敛情况。
/// 超参数偏小（dModel=8、单层等），方便初学者快速跑通并观察 loss 下降。
/// </summary>
static class Program
{
    // ---------- 序列与数据 ----------
    /// <summary>序列最大长度。格式为 [SOS] + 最多 10 个数字 + [EOS] + PAD 填充至该长度。</summary>
    const int MaxLen = 14;

    /// <summary>每个 batch 的样本数。越大则 loss 曲线越平滑，但单步计算量增加。</summary>
    const int BatchSize = 64;

    // ---------- 训练轮数与优化器 ----------
    /// <summary>训练轮数（一个 epoch = 完整遍历一遍训练集）。反向传播下约 80 轮即可收敛。</summary>
    const int Epochs = 80;

    /// <summary>SGD 学习率，用于反向传播模式。过大易发散，过小收敛慢。</summary>
    const float LrSgd = 0.1f;

    /// <summary>SPSA 模式下的学习率（仅当命令行传入 "spsa" 时使用）。</summary>
    const float LrSpsa = 0.05f;

    /// <summary>SPSA 的扰动步长 epsilon，用于数值估计梯度方向。</summary>
    const float Eps = 0.03f;

    /// <summary>模型权重与配置的保存目录（相对可执行文件）。</summary>
    const string ModelDir = "SavedModel";

    static void Main(string[] args)
    {
        Run();
        //Train();
    }

    static void Run()
    {
        string dir = ModelDir;
        Console.WriteLine("--- 推理示例（加载已保存模型） ---\n");
        // 构造几条示例：输入为 0~9 的整数序列，输出为每个元素加 1（模 10）
        var examples = new[]
        {
            new[] { 1, 2, 3 },
            new[] { 5, 6, 7, 8 },
            new[] { 0, 9, 4 },
        };
        InferenceExample.Run(examples,dir);
        Console.WriteLine("按任意键退出...");
        Console.ReadKey();
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
