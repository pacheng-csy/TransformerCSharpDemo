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

    public EncoderLayer(int dModel, int numHeads, int dFf)
    {
        _selfAttn = new MultiHeadAttention(dModel, numHeads);
        _ffn = new PositionwiseFeedForward(dModel, dFf);
        _norm1 = new LayerNorm(dModel);
        _norm2 = new LayerNorm(dModel);
    }

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

    private static float[][][] Add3D(float[][][] a, float[][][] b)
    {
        var c = new float[a.Length][][];
        for (int i = 0; i < a.Length; i++)
            c[i] = MatrixHelper.Add(a[i], b[i]);
        return c;
    }

    private static void AddInPlace3(float[][][] target, float[][][] b)
    {
        for (int i = 0; i < target.Length; i++)
            MatrixHelper.AddInPlace(target[i], b[i]);
    }

    public MultiHeadAttention SelfAttn => _selfAttn;
    public PositionwiseFeedForward Ffn => _ffn;
}
