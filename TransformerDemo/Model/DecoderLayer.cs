using System;

namespace TransformerDemo;

/// <summary>
/// 解码器层：Masked Multi-Head Self-Attention + Add & Norm + Cross-Attention (Q 来自解码器, K/V 来自编码器) + Add & Norm + FFN + Add & Norm。
/// Causal mask 在 Self-Attention 中应用，防止看到未来位置。
/// </summary>
public class DecoderLayer
{
    private readonly MultiHeadAttention _selfAttn;
    private readonly MultiHeadAttention _crossAttn;
    private readonly PositionwiseFeedForward _ffn;
    private readonly LayerNorm _norm1, _norm2, _norm3;

    public DecoderLayer(int dModel, int numHeads, int dFf)
    {
        _selfAttn = new MultiHeadAttention(dModel, numHeads);
        _crossAttn = new MultiHeadAttention(dModel, numHeads);
        _ffn = new PositionwiseFeedForward(dModel, dFf);
        _norm1 = new LayerNorm(dModel);
        _norm2 = new LayerNorm(dModel);
        _norm3 = new LayerNorm(dModel);
    }

    public float[][][] Forward(float[][][] decInput, float[][][] encOutput, bool[][][]? decPaddingMask, bool[][][]? encPaddingMask)
    {
        int decLen = decInput[0].Length;
        bool[][] causalMask = MatrixHelper.CreateCausalMask(decLen);
        bool[][][]? selfMask = decPaddingMask != null
            ? MatrixHelper.CombinePaddingAndCausal(decPaddingMask, causalMask)
            : null;
        for (int b = 0; b < (selfMask?.Length ?? 0); b++)
            for (int i = 0; i < decLen; i++)
                for (int j = 0; j < decLen; j++)
                    if (!causalMask[i][j]) selfMask![b][i][j] = false;

        if (selfMask == null)
        {
            selfMask = new bool[decInput.Length][][];
            for (int b = 0; b < decInput.Length; b++)
                selfMask[b] = causalMask;
        }

        // Masked Self-Attention + Add & Norm
        var selfOut = _selfAttn.Forward(decInput, decInput, decInput, selfMask);
        var x1 = Add3D(decInput, selfOut);
        x1 = _norm1.Forward3(x1);

        // Cross-Attention: Q 来自解码器, K/V 来自编码器
        var crossOut = _crossAttn.Forward(x1, encOutput, encOutput, encPaddingMask);
        var x2 = Add3D(x1, crossOut);
        x2 = _norm2.Forward3(x2);

        // FFN + Add & Norm
        var ffnOut = _ffn.Forward(x2);
        var x3 = Add3D(x2, ffnOut);
        return _norm3.Forward3(x3);
    }

    /// <summary>反向：dLdOut 与输出同形状。返回 (dLdDecInput, dLdEncOutput)。</summary>
    public (float[][][] dLdDecInput, float[][][] dLdEncOutput) Backward(float[][][] dLdOut)
    {
        var dLdX3Norm = _norm3.Backward3(dLdOut);
        var dLdX2 = MatrixHelper.Copy3(dLdX3Norm);
        var dLdFfn = _ffn.Backward(dLdX3Norm);
        AddInPlace3(dLdX2, dLdFfn);

        var dLdX2Norm = _norm2.Backward3(dLdX2);
        var dLdX1 = MatrixHelper.Copy3(dLdX2Norm);
        var (dLdQ2, dLdK2, dLdV2) = _crossAttn.Backward(dLdX2Norm);
        AddInPlace3(dLdX1, dLdQ2);
        var dLdEncOutput = Add3D(dLdK2, dLdV2);

        var dLdX1Norm = _norm1.Backward3(dLdX1);
        var (dLdQ, dLdK, dLdV) = _selfAttn.Backward(dLdX1Norm);
        var dLdDecInput = Add3D(dLdX1Norm, Add3D(dLdQ, Add3D(dLdK, dLdV)));
        return (dLdDecInput, dLdEncOutput);
    }

    public void ZeroGrad()
    {
        _selfAttn.ZeroGrad();
        _crossAttn.ZeroGrad();
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
    public MultiHeadAttention CrossAttn => _crossAttn;
    public PositionwiseFeedForward Ffn => _ffn;
}
