using System;

namespace TransformerDemo;

/// <summary>
/// 编码器层：Multi-Head Self-Attention + Add & Norm + FFN + Add & Norm。
/// 输入输出形状均为 (batch, seqLen, d_model)。
/// </summary>
public class EncoderLayer
{
    private readonly MultiHeadAttention _selfAttn;
    private readonly PositionwiseFeedForward _ffn;
    private readonly LayerNorm _norm1, _norm2;

    /// <summary>
    /// 构造一个标准 Transformer 编码器层。
    /// </summary>
    /// <param name="dModel">隐藏维度 d_model</param>
    /// <param name="numHeads">自注意力头数</param>
    /// <param name="dFf">前馈网络隐藏层维度</param>
    public EncoderLayer(int dModel, int numHeads, int dFf)
    {
        _selfAttn = new MultiHeadAttention(dModel, numHeads);
        _ffn = new PositionwiseFeedForward(dModel, dFf);
        _norm1 = new LayerNorm(dModel);
        _norm2 = new LayerNorm(dModel);
    }

    /// <summary>
    /// 前向传播：Self-Attention + Add &amp; Norm + FFN + Add &amp; Norm。
    /// </summary>
    /// <param name="x">输入张量 (batch, seqLen, d_model)</param>
    /// <param name="paddingMask">Padding 掩码 (batch, seqLen, seqLen)，true 表示允许关注，false 为屏蔽；可为 null</param>
    /// <returns>同形状的编码器输出</returns>
    public float[][][] Forward(float[][][] x, bool[][][]? paddingMask)
    {
        // Self-Attention + Add & Norm
        var attnOut = _selfAttn.Forward(x, x, x, paddingMask);
        var x1 = Add3D(x, attnOut);
        x1 = _norm1.Forward3(x1);

        // FFN + Add & Norm
        var ffnOut = _ffn.Forward(x1);
        var x2 = Add3D(x1, ffnOut);
        return _norm2.Forward3(x2);
    }

    /// <summary>反向：dLdOut 与输出同形状，返回 dLdX。</summary>
    public float[][][] Backward(float[][][] dLdOut)
    {
        var dLdX2Norm = _norm2.Backward3(dLdOut);
        var dLdX1 = MatrixHelper.Copy3(dLdX2Norm);
        var dLdFfn = _ffn.Backward(dLdX2Norm);
        AddInPlace3(dLdX1, dLdFfn);
        var dLdX1Norm = _norm1.Backward3(dLdX1);
        var (dLdQ, dLdK, dLdV) = _selfAttn.Backward(dLdX1Norm);
        var dLdX = Add3D(dLdX1Norm, Add3D(dLdQ, Add3D(dLdK, dLdV)));
        return dLdX;
    }

    public void ZeroGrad()
    {
        _selfAttn.ZeroGrad();
        _ffn.ZeroGrad();
    }

    /// <summary>
    /// 逐元素相加两个 3D 张量：c[b] = a[b] + b[b]。
    /// </summary>
    private static float[][][] Add3D(float[][][] a, float[][][] b)
    {
        var c = new float[a.Length][][];
        for (int i = 0; i < a.Length; i++)
            c[i] = MatrixHelper.Add(a[i], b[i]);
        return c;
    }

    /// <summary>
    /// 就地对 3D 张量做逐元素加法：target[b] += b[b]。
    /// </summary>
    private static void AddInPlace3(float[][][] target, float[][][] b)
    {
        for (int i = 0; i < target.Length; i++)
            MatrixHelper.AddInPlace(target[i], b[i]);
    }

    public MultiHeadAttention SelfAttn => _selfAttn;
    public PositionwiseFeedForward Ffn => _ffn;
}
