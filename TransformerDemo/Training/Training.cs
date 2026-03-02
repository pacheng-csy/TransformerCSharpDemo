using System;
using System.Collections.Generic;
using System.Linq;

namespace TransformerDemo;

/// <summary>
/// 训练与验证模块：负责
/// - 计算损失（loss）；
/// - 用一种“简化版”的优化算法（SPSA）更新参数；
/// - 完成按 epoch 的训练/验证循环。
///
/// 【为什么不用真正的反向传播？】
/// 真正的 Transformer 训练需要对每一层都手写梯度推导和反向传播代码，对入门教学来说工作量太大、细节太多。
/// 这里采用 SPSA（Simultaneous Perturbation Stochastic Approximation，同时扰动随机逼近）：
/// - 每次随机出一个方向 d，在参数空间沿着 +eps*d 和 -eps*d 各走一步，得到 loss_plus 和 loss_minus；
/// - 用 (loss_plus - loss_minus) / (2*eps) 近似“在方向 d 上的导数”；
/// - 再反过来用这个方向对所有参数做一次统一更新。
/// 这样我们只需要写“前向计算”和“如何展开/回填参数”两件事，就能看到 loss 下降的全过程，便于初学者理解训练流程。
/// </summary>
public static class Training
{
    private static readonly Random Rng = new(123);

    /// <summary>计算一个 batch 的交叉熵损失，忽略 [PAD] 位置。</summary>
    public static float ComputeLoss(
        TransformerModel model,
        int[][] encIds, int[][] decIds,
        int[] encValidLengths, int[] decValidLengths,
        int[][] targetIds)
    {
        var logits = model.Forward(encIds, decIds, encValidLengths, decValidLengths);
        int batch = logits.Length, decLen = logits[0].Length, vocabSize = logits[0][0].Length;
        int N = batch * decLen;
        var logitsFlat = new float[N][];
        var targetFlat = new int[N];
        var paddingMask = new bool[N][];
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < decLen; s++)
            {
                int n = b * decLen + s;
                logitsFlat[n] = logits[b][s];
                targetFlat[n] = targetIds[b][s];
                paddingMask[n] = new[] { targetIds[b][s] != Vocabulary.PadId };
            }
        }
        return MatrixHelper.CrossEntropyLoss(logitsFlat, targetFlat, paddingMask);
    }

    /// <summary>SPSA 单步：随机方向 d，估计梯度后 params -= lr * 估计梯度。</summary>
    public static void SpsaStep(
        TransformerModel model,
        int[][] encIds, int[][] decIds,
        int[] encValidLengths, int[] decValidLengths,
        int[][] targetIds,
        float lr, float eps)
    {
        float[] params_ = model.GetParametersFlat();
        int n = params_.Length;
        var d = new float[n];
        for (int i = 0; i < n; i++)
            d[i] = (float)(Rng.NextDouble() * 2 - 1);
        float norm = 0;
        for (int i = 0; i < n; i++)
            norm += d[i] * d[i];
        norm = (float)Math.Sqrt(norm);
        if (norm > 0)
            for (int i = 0; i < n; i++)
                d[i] /= norm;

        for (int i = 0; i < n; i++)
            params_[i] += eps * d[i];
        model.SetParametersFromFlat(params_);
        float lossPlus = ComputeLoss(model, encIds, decIds, encValidLengths, decValidLengths, targetIds);

        for (int i = 0; i < n; i++)
            params_[i] -= 2 * eps * d[i];
        model.SetParametersFromFlat(params_);
        float lossMinus = ComputeLoss(model, encIds, decIds, encValidLengths, decValidLengths, targetIds);

        float gradEst = (lossPlus - lossMinus) / (2 * eps);
        for (int i = 0; i < n; i++)
            params_[i] += eps * d[i];
        for (int i = 0; i < n; i++)
            params_[i] -= lr * gradEst * d[i];
        model.SetParametersFromFlat(params_);
    }

    /// <summary>训练一个 epoch：遍历所有 batch，每 batch 做一次 SPSA 更新。</summary>
    public static float TrainEpoch(
        TransformerModel model,
        IReadOnlyList<DataGenerator.Sample> trainData,
        int maxLen, int batchSize, Vocabulary vocab,
        float lr, float eps)
    {
        float totalLoss = 0;
        int batchCount = 0;
        foreach (var (inputIds, targetIds, validLengths) in DataGenerator.Batches(trainData, maxLen, batchSize, vocab))
        {
            SpsaStep(model, inputIds, targetIds, validLengths, validLengths, targetIds, lr, eps);
            totalLoss += ComputeLoss(model, inputIds, targetIds, validLengths, validLengths, targetIds);
            batchCount++;
        }
        return batchCount > 0 ? totalLoss / batchCount : 0;
    }

    /// <summary>反向传播 + SGD 单步：前向 → 算 loss → CE 梯度 → Backward → 参数 -= lr * grad。返回本 batch 的 loss。</summary>
    public static float TrainStepBackprop(
        TransformerModel model,
        int[][] encIds, int[][] decIds,
        int[] encValidLengths, int[] decValidLengths,
        int[][] targetIds,
        float lr)
    {
        model.ZeroGrad();
        var logits = model.Forward(encIds, decIds, encValidLengths, decValidLengths);
        int batch = logits.Length, decLen = logits[0].Length, vocabSize = logits[0][0].Length;
        int N = batch * decLen;
        var logitsFlat = new float[N][];
        var targetFlat = new int[N];
        var paddingMask = new bool[N][];
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < decLen; s++)
            {
                int n = b * decLen + s;
                logitsFlat[n] = logits[b][s];
                targetFlat[n] = targetIds[b][s];
                paddingMask[n] = new[] { targetIds[b][s] != Vocabulary.PadId };
            }
        }
        float loss = MatrixHelper.CrossEntropyLoss(logitsFlat, targetFlat, paddingMask);
        var dLdLogitsFlat = MatrixHelper.CrossEntropyLossBackward(logitsFlat, targetFlat, paddingMask, out _);
        var dLdLogits3D = new float[batch][][];
        for (int b = 0; b < batch; b++)
        {
            dLdLogits3D[b] = new float[decLen][];
            for (int s = 0; s < decLen; s++)
                dLdLogits3D[b][s] = dLdLogitsFlat[b * decLen + s];
        }
        model.Backward(dLdLogits3D, encIds, decIds);
        var parameters = model.GetAllParameters();
        var gradients = model.GetAllGradients();
        for (int i = 0; i < parameters.Count; i++)
        {
            var p = parameters[i];
            var g = gradients[i];
            for (int r = 0; r < p.Length; r++)
                for (int c = 0; c < p[r].Length; c++)
                    p[r][c] -= lr * g[r][c];
        }
        return loss;
    }

    /// <summary>训练一个 epoch：反向传播 + SGD（每 batch 一步）。</summary>
    public static float TrainEpochBackprop(
        TransformerModel model,
        IReadOnlyList<DataGenerator.Sample> trainData,
        int maxLen, int batchSize, Vocabulary vocab,
        float lr)
    {
        float totalLoss = 0;
        int batchCount = 0;
        foreach (var (inputIds, targetIds, validLengths) in DataGenerator.Batches(trainData, maxLen, batchSize, vocab))
        {
            totalLoss += TrainStepBackprop(model, inputIds, targetIds, validLengths, validLengths, targetIds, lr);
            batchCount++;
        }
        return batchCount > 0 ? totalLoss / batchCount : 0;
    }

    /// <summary>验证：不更新参数，只计算平均 loss。</summary>
    public static float Validate(
        TransformerModel model,
        IReadOnlyList<DataGenerator.Sample> validData,
        int maxLen, int batchSize, Vocabulary vocab)
    {
        float totalLoss = 0;
        int batchCount = 0;
        foreach (var (inputIds, targetIds, validLengths) in DataGenerator.Batches(validData, maxLen, batchSize, vocab))
        {
            totalLoss += ComputeLoss(model, inputIds, targetIds, validLengths, validLengths, targetIds);
            batchCount++;
        }
        return batchCount > 0 ? totalLoss / batchCount : 0;
    }

    /// <summary>训练一个 epoch（已 token 化样本）：使用 BatchesTokenized，反向传播 + SGD。</summary>
    public static float TrainEpochBackpropTokenized(
        TransformerModel model,
        IReadOnlyList<DataGenerator.Sample> trainData,
        int maxLen, int batchSize, float lr)
    {
        float totalLoss = 0;
        int batchCount = 0;
        foreach (var (inputIds, targetIds, validLengths) in DataGenerator.BatchesTokenized(trainData, maxLen, batchSize))
        {
            totalLoss += TrainStepBackprop(model, inputIds, targetIds, validLengths, validLengths, targetIds, lr);
            batchCount++;
        }
        return batchCount > 0 ? totalLoss / batchCount : 0;
    }

    /// <summary>验证（已 token 化样本）：使用 BatchesTokenized，只计算平均 loss。</summary>
    public static float ValidateTokenized(
        TransformerModel model,
        IReadOnlyList<DataGenerator.Sample> validData,
        int maxLen, int batchSize)
    {
        float totalLoss = 0;
        int batchCount = 0;
        foreach (var (inputIds, targetIds, validLengths) in DataGenerator.BatchesTokenized(validData, maxLen, batchSize))
        {
            totalLoss += ComputeLoss(model, inputIds, targetIds, validLengths, validLengths, targetIds);
            batchCount++;
        }
        return batchCount > 0 ? totalLoss / batchCount : 0;
    }

    /// <summary>Token 级准确率（忽略 PAD）：预测 id 与 target 一致的比例。</summary>
    public static float TokenAccuracy(
        TransformerModel model,
        int[][] encIds, int[][] decIds,
        int[] encValidLengths, int[] decValidLengths,
        int[][] targetIds)
    {
        var logits = model.Forward(encIds, decIds, encValidLengths, decValidLengths);
        int correct = 0, total = 0;
        for (int b = 0; b < logits.Length; b++)
            for (int s = 0; s < logits[0].Length; s++)
            {
                if (targetIds[b][s] == Vocabulary.PadId) continue;
                total++;
                var pred = MatrixHelper.ArgmaxLastDim(new[] { logits[b][s] });
                if (pred[0] == targetIds[b][s]) correct++;
            }
        return total > 0 ? (float)correct / total : 0;
    }
}
