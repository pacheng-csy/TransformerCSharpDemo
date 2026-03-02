using System;

namespace TransformerDemo;

/// <summary>
/// 缩放点积注意力（Scaled Dot-Product Attention）
///
/// 【Q、K、V 的含义（可以类比“查字典”）】
/// - Query (Q)：查询，表示“我现在想找什么信息”；对应当前序列位置的向量；
/// - Key (K)：键，表示“我能被怎样的 Query 找到”；对应所有位置的向量；
/// - Value (V)：值，表示“如果我被找到，要把什么内容贡献给别人”；也对应所有位置的向量。
/// 算法可以理解为：
///   1. 对于每个 Query，和所有 Key 做点积，得到相似度分数 Score = Q * K^T；
///   2. 对每一行 Score 做 Softmax，得到“关注每个位置的概率”；
///   3. 用这些概率对 Value 做加权求和，得到当前 Query 的输出向量。
/// 数学形式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
///
/// 【为什么要除以 sqrt(d_k)（缩放因子的作用）】
/// 当向量维度 d_k 很大时，QK^T 的值通常也会非常大，Softmax 很容易饱和（接近 0 或 1），
/// 这会让梯度变得很小、训练变慢。除以 sqrt(d_k) 可以把分数缩放到一个比较合适的范围，
/// 让 Softmax 不那么极端，从而训练更稳定。
/// </summary>
public static class ScaledDotProductAttention
{
    /// <summary>
    /// 前向：Q、K、V 形状均为 (batch, seqLen, d_k)，mask 形状 (batch, seqLen, seqLen)，false 表示屏蔽。
    /// 返回 (output, attentionWeights)，output 形状 (batch, seqLen, d_k)，attentionWeights (batch, seqLen, seqLen)。
    /// </summary>
    public static (float[][][] output, float[][][] attentionWeights) Forward(
        float[][][] q, float[][][] k, float[][][] v,
        bool[][][]? mask = null)
    {
        int batch = q.Length, seqQ = q[0].Length, seqK = k[0].Length, dk = q[0][0].Length;
        float scale = (float)(1.0 / Math.Sqrt(dk));

        // scores[b] = Q[b] * K[b]^T  => (seqQ, seqK)
        var scores = new float[batch][][];
        for (int b = 0; b < batch; b++)
        {
            var kt = MatrixHelper.Transpose(k[b]);
            scores[b] = MatrixHelper.Multiply(q[b], kt);
            for (int i = 0; i < seqQ; i++)
                for (int j = 0; j < seqK; j++)
                    scores[b][i][j] *= scale;
        }

        // Softmax + mask（mask 为 false 的位置在 Softmax 前视为 -inf）
        var attn = MatrixHelper.SoftmaxLastDim3(scores, mask);

        // output[b] = attn[b] * V[b]  => (seqQ, dk)
        var output = new float[batch][][];
        for (int b = 0; b < batch; b++)
            output[b] = MatrixHelper.Multiply(attn[b], v[b]);

        return (output, attn);
    }

    /// <summary>
    /// 反向传播：给定 dLdOutput (batch, seqQ, d_k)，以及前向缓存的 Q,K,V, attnWeights, scale。
    /// 返回 (dLdQ, dLdK, dLdV)，形状与 Q,K,V 相同。
    /// </summary>
    public static (float[][][] dLdQ, float[][][] dLdK, float[][][] dLdV) Backward(
        float[][][] dLdOutput,
        float[][][] q, float[][][] k, float[][][] v,
        float[][][] attnWeights,
        float scale)
    {
        int batch = q.Length, seqQ = q[0].Length, seqK = k[0].Length, dk = q[0][0].Length;
        var dLdQ = MatrixHelper.Zeros3(batch, seqQ, dk);
        var dLdK = MatrixHelper.Zeros3(batch, seqK, dk);
        var dLdV = MatrixHelper.Zeros3(batch, seqK, dk);

        for (int b = 0; b < batch; b++)
        {
            // output = P * V  =>  dLdP = dLdOutput * V^T,  dLdV = P^T * dLdOutput
            var vt = MatrixHelper.Transpose(v[b]);
            var dLdP = MatrixHelper.Multiply(dLdOutput[b], vt);
            var pt = MatrixHelper.Transpose(attnWeights[b]);
            var dLdVb = MatrixHelper.Multiply(pt, dLdOutput[b]);
            dLdV[b] = dLdVb;

            // P = softmax(S)  =>  dLdS = softmax_backward(dLdP, P)
            var dLdS = MatrixHelper.SoftmaxLastDimBackward(dLdP, attnWeights[b]);

            // S = Q * K^T * scale  =>  dLdQ = scale * dLdS * K,  dLdK = scale * dLdS^T * Q
            for (int i = 0; i < seqQ; i++)
                for (int j = 0; j < seqK; j++)
                    dLdS[i][j] *= scale;
            var dLdQb = MatrixHelper.Multiply(dLdS, k[b]);
            var dLdKb = MatrixHelper.Multiply(MatrixHelper.Transpose(dLdS), q[b]);
            dLdQ[b] = dLdQb;
            dLdK[b] = dLdKb;
        }
        return (dLdQ, dLdK, dLdV);
    }
}
