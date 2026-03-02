using System;
using System.Collections.Generic;
using System.Linq;

namespace TransformerDemo;

/// <summary>
/// 序列到序列数据生成与预处理（相当于我们给 Transformer 设计的一份“玩具作业”）：
/// - 原始任务：输入是随机长度 5~10 的整数序列（0~9），输出是“每个元素加 1”的序列（模 10，仍为 0~9）；
/// - 训练集：共 1000 条样本；验证集：共 200 条样本，全部在内存中随机生成；
/// - 预处理步骤：
///   1. 先把每个整数用 Vocabulary 映射成 token id；
///   2. 在序列开头加 [SOS]，结尾加 [EOS]，告诉模型“序列从这里开始/到这里结束”；
///   3. 不足固定长度的部分用 [PAD] 填充，方便后续用矩阵批量计算。
/// 通过这个类，你可以清楚地看到“原始整数序列”是如何一步步变成 Transformer 能吃的“token 序列”的。
/// </summary>
public static class DataGenerator
{
    /// <summary>生成样本时序列的最小长度。</summary>
    public const int MinSeqLen = 5;
    /// <summary>生成样本时序列的最大长度。</summary>
    public const int MaxSeqLen = 10;
    /// <summary>训练集样本数量。</summary>
    public const int TrainCount = 1000;
    /// <summary>验证集样本数量。</summary>
    public const int ValidCount = 200;

    /// <summary>单条样本：原始输入序列、目标序列（已含加一后的数字，未加特殊 token）</summary>
    public record Sample(int[] Input, int[] Target);

    /// <summary>用于随机生成样本的伪随机数发生器，固定种子保证结果可复现。</summary>
    private static readonly Random Rng = new(42);

    /// <summary>生成一条随机样本：输入长度在 [MinSeqLen, MaxSeqLen]，每个元素 0~9；目标为每元素+1（模10）</summary>
    /// <summary>
    /// 随机生成一条“数字加一”任务样本。
    /// </summary>
    /// <returns>
    /// 返回包含原始输入序列和目标序列的 <see cref="Sample"/>：
    /// 输入为长度在 [MinSeqLen, MaxSeqLen] 内的 0~9 序列，目标为对每一位做 +1 (mod 10)。
    /// </returns>
    public static Sample GenerateOne()
    {
        int len = Rng.Next(MinSeqLen, MaxSeqLen + 1);
        var input = new int[len];
        var target = new int[len];
        for (int i = 0; i < len; i++)
        {
            input[i] = Rng.Next(0, 10);
            target[i] = (input[i] + 1) % 10;
        }
        return new Sample(input, target);
    }

    /// <summary>生成训练集与验证集</summary>
    /// <summary>
    /// 批量生成训练集与验证集样本。
    /// </summary>
    /// <returns>
    /// 一个二元组：(train, valid)，其中 train/valid 都是样本列表，
    /// 大小分别为 <see cref="TrainCount"/> 和 <see cref="ValidCount"/>。
    /// </returns>
    public static (List<Sample> train, List<Sample> valid) GenerateAll()
    {
        var train = new List<Sample>();
        for (int i = 0; i < TrainCount; i++)
            train.Add(GenerateOne());
        var valid = new List<Sample>();
        for (int i = 0; i < ValidCount; i++)
            valid.Add(GenerateOne());
        return (train, valid);
    }

    /// <summary>
    /// 预处理：为输入添加 [SOS] 和 [EOS]，用 [PAD] 填充到 maxLen。
    /// 目标序列格式： [SOS] + target数字序列 + [EOS]，再填充到 maxLen。
    /// 返回：(inputIds, targetIds, validLengths)
    /// validLengths[b] = 有效长度（不含 PAD），用于构造 attention mask。
    /// </summary>
    public static (int[][] inputIds, int[][] targetIds, int[] validLengths) Preprocess(
        IReadOnlyList<Sample> samples,
        int maxLen,
        Vocabulary vocab)
    {
        int n = samples.Count;
        var inputIds = new int[n][];
        var targetIds = new int[n][];
        var validLengths = new int[n];

        for (int b = 0; b < n; b++)
        {
            var s = samples[b];
            // 输入: [SOS] + input + [EOS] + PAD
            int inLen = 1 + s.Input.Length + 1;
            validLengths[b] = inLen;
            var inp = new List<int> { Vocabulary.SosId };
            foreach (int d in s.Input)
                inp.Add(vocab.EncodeDigit(d));
            inp.Add(Vocabulary.EosId);
            while (inp.Count < maxLen)
                inp.Add(Vocabulary.PadId);
            inputIds[b] = inp.Take(maxLen).ToArray();

            // 目标: [SOS] + target + [EOS] + PAD（解码时预测下一个 token，所以 target 与 input 错位一格也可；这里用整序列便于计算 loss）
            var tgt = new List<int> { Vocabulary.SosId };
            foreach (int d in s.Target)
                tgt.Add(vocab.EncodeDigit(d));
            tgt.Add(Vocabulary.EosId);
            while (tgt.Count < maxLen)
                tgt.Add(Vocabulary.PadId);
            targetIds[b] = tgt.Take(maxLen).ToArray();
        }

        return (inputIds, targetIds, validLengths);
    }

    /// <summary>
    /// 按 batch 迭代：返回 (inputIds, targetIds, validLengths) 每批。
    /// targetIds 用于 loss：预测的是每个位置的下一 token，这里用 targetIds 与 logits 对齐（即 decoder 输出与 target 同长度，算 CE 时用 target 对应位置）。
    /// </summary>
    public static IEnumerable<(int[][] inputIds, int[][] targetIds, int[] validLengths)> Batches(
        IReadOnlyList<Sample> samples,
        int maxLen,
        int batchSize,
        Vocabulary vocab)
    {
        var (inputIds, targetIds, validLengths) = Preprocess(samples, maxLen, vocab);
        for (int start = 0; start < inputIds.Length; start += batchSize)
        {
            int count = Math.Min(batchSize, inputIds.Length - start);
            var batchInput = new int[count][];
            var batchTarget = new int[count][];
            var batchValid = new int[count];
            for (int i = 0; i < count; i++)
            {
                batchInput[i] = inputIds[start + i];
                batchTarget[i] = targetIds[start + i];
                batchValid[i] = validLengths[start + i];
            }
            yield return (batchInput, batchTarget, batchValid);
        }
    }

    /// <summary>
    /// 预处理“已 token 化”的样本：Sample.Input / Target 已是 token id 序列（仅内容），此处添加 [SOS]/[EOS] 并用 [PAD] 填充到 maxLen。
    /// 与 Preprocess 输出格式一致，供三国问答等文本任务使用。
    /// </summary>
    public static (int[][] inputIds, int[][] targetIds, int[] validLengths) PreprocessTokenized(
        IReadOnlyList<Sample> samples,
        int maxLen)
    {
        int n = samples.Count;
        var inputIds = new int[n][];
        var targetIds = new int[n][];
        var validLengths = new int[n];

        for (int b = 0; b < n; b++)
        {
            var s = samples[b];
            var inp = new List<int> { Vocabulary.SosId };
            foreach (int id in s.Input)
                inp.Add(id);
            inp.Add(Vocabulary.EosId);
            validLengths[b] = inp.Count;
            while (inp.Count < maxLen)
                inp.Add(Vocabulary.PadId);
            inputIds[b] = inp.Take(maxLen).ToArray();

            var tgt = new List<int> { Vocabulary.SosId };
            foreach (int id in s.Target)
                tgt.Add(id);
            tgt.Add(Vocabulary.EosId);
            while (tgt.Count < maxLen)
                tgt.Add(Vocabulary.PadId);
            targetIds[b] = tgt.Take(maxLen).ToArray();
        }

        return (inputIds, targetIds, validLengths);
    }

    /// <summary>
    /// 按 batch 迭代已 token 化样本，返回 (inputIds, targetIds, validLengths) 每批。
    /// </summary>
    public static IEnumerable<(int[][] inputIds, int[][] targetIds, int[] validLengths)> BatchesTokenized(
        IReadOnlyList<Sample> samples,
        int maxLen,
        int batchSize)
    {
        var (inputIds, targetIds, validLengths) = PreprocessTokenized(samples, maxLen);
        for (int start = 0; start < inputIds.Length; start += batchSize)
        {
            int count = Math.Min(batchSize, inputIds.Length - start);
            var batchInput = new int[count][];
            var batchTarget = new int[count][];
            var batchValid = new int[count];
            for (int i = 0; i < count; i++)
            {
                batchInput[i] = inputIds[start + i];
                batchTarget[i] = targetIds[start + i];
                batchValid[i] = validLengths[start + i];
            }
            yield return (batchInput, batchTarget, batchValid);
        }
    }
}
