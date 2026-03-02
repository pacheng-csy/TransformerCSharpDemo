using System;
using System.Collections.Generic;

namespace TransformerDemo;

/// <summary>
/// 模型质量评估模块（单独文件，便于教学演示）。
/// 在这里我们不会更新参数，只会在验证集上做“考试”，给出几个直观的指标：
/// - 验证集平均 loss（越低越好）；
/// - Token 级准确率：所有非 [PAD] 位置上，预测 token 与目标 token 完全相同的比例；
/// - 序列级准确率：一整条序列的所有非 [PAD] 位置都预测正确，才算这条序列“完全正确”的比例。
/// </summary>
public static class Evaluation
{
    /// <summary>
    /// 在完整验证集上评估当前模型质量，并打印结果。
    /// 调用时不会修改模型参数，只做前向计算。
    /// </summary>
    public static void EvaluateOnValidation(
        TransformerModel model,
        IReadOnlyList<DataGenerator.Sample> validData,
        int maxLen,
        int batchSize,
        Vocabulary vocab)
    {
        // 1. 先用 Training.Validate 计算一遍“整体验证 loss”
        float validLoss = Training.Validate(model, validData, maxLen, batchSize, vocab);

        // 2. 再按 batch 计算 token 级准确率和序列级准确率
        float sumTokenAcc = 0;
        float sumSeqAcc = 0;
        int batchCount = 0;

        foreach (var (inputIds, targetIds, validLengths) in DataGenerator.Batches(validData, maxLen, batchSize, vocab))
        {
            // 2.1 Token 级准确率（调用已实现的 Training.TokenAccuracy）
            float tokenAcc = Training.TokenAccuracy(
                model,
                inputIds,
                targetIds,
                validLengths,
                validLengths,
                targetIds);
            sumTokenAcc += tokenAcc;

            // 2.2 序列级准确率：一整条序列所有非 PAD 位置都预测正确才算 1 条对
            float seqAcc = ComputeSequenceAccuracy(model, inputIds, targetIds, validLengths);
            sumSeqAcc += seqAcc;

            batchCount++;
        }

        float avgTokenAcc = batchCount > 0 ? sumTokenAcc / batchCount : 0f;
        float avgSeqAcc = batchCount > 0 ? sumSeqAcc / batchCount : 0f;

        Console.WriteLine("=== 模型质量评估（完整验证集） ===");
        Console.WriteLine($"验证集平均 Loss        : {validLoss:F4}");
        Console.WriteLine($"Token 级准确率（平均） : {avgTokenAcc * 100:F2}%");
        Console.WriteLine($"序列级准确率（平均）   : {avgSeqAcc * 100:F2}%");
        Console.WriteLine();
    }

    /// <summary>
    /// 计算一个 batch 的“序列级准确率”：只要序列中有一个非 PAD 位置预测错，这条序列就算错。
    /// </summary>
    private static float ComputeSequenceAccuracy(
        TransformerModel model,
        int[][] encIds,
        int[][] decIds,
        int[] validLengths)
    {
        // 这里我们让解码器的输入 decIds 直接等于目标序列（teacher forcing），
        // 和训练时期的用法保持一致，只是这次不更新参数。
        var logits = model.Forward(encIds, decIds, validLengths, validLengths);
        int batch = logits.Length;
        int decLen = logits[0].Length;

        int correctSeq = 0;

        for (int b = 0; b < batch; b++)
        {
            bool allCorrect = true;
            for (int s = 0; s < decLen; s++)
            {
                int targetId = decIds[b][s];
                // 忽略 PAD 位置
                if (targetId == Vocabulary.PadId)
                    continue;

                // 取该位置上概率最大的类别
                int best = 0;
                for (int c = 1; c < logits[0][0].Length; c++)
                    if (logits[b][s][c] > logits[b][s][best]) best = c;

                if (best != targetId)
                {
                    allCorrect = false;
                    break;
                }
            }

            if (allCorrect)
                correctSeq++;
        }

        return batch > 0 ? (float)correctSeq / batch : 0f;
    }
}

