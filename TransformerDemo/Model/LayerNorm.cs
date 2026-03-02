using System;

namespace TransformerDemo;

/// <summary>
/// LayerNorm：对最后一维做归一化 (x - mean) / sqrt(var + eps)。
/// 用于 Add & Norm 子层，稳定训练。
/// </summary>
public class LayerNorm
{
    private readonly float _eps;
    private readonly int _dModel;
    private float[][][]? _lastX;
    private float[][]? _lastMean;
    private float[][]? _lastVar;

    /// <summary>
    /// 构造 LayerNorm 模块，内部不含可学习参数，仅保存维度与数值稳定项。
    /// </summary>
    /// <param name="dModel">最后一维大小（特征维度）</param>
    /// <param name="eps">用于防止除零的极小常数</param>
    public LayerNorm(int dModel, float eps = 1e-6f)
    {
        _dModel = dModel;
        _eps = eps;
    }

    /// <summary>Forward: x (rows, d_model)。每行归一化，返回同形状。</summary>
    public float[][] Forward(float[][] x)
    {
        return MatrixHelper.LayerNormForward(x, _eps, out _, out _);
    }

    /// <summary>3D: 对每个 (b,s) 的 d_model 维做 LayerNorm。x (B, seq, d_model)。</summary>
    public float[][][] Forward3(float[][][] x)
    {
        int B = x.Length, seq = x[0].Length;
        _lastX = MatrixHelper.Copy3(x);
        _lastMean = new float[B][];
        _lastVar = new float[B][];
        var out_ = new float[B][][];
        for (int b = 0; b < B; b++)
        {
            out_[b] = MatrixHelper.LayerNormForward(x[b], _eps, out _lastMean[b], out _lastVar[b]);
        }
        return out_;
    }

    /// <summary>3D 反向：dLdOut 与输出同形状，返回 dLdX。</summary>
    public float[][][] Backward3(float[][][] dLdOut)
    {
        if (_lastX == null || _lastMean == null || _lastVar == null)
            throw new InvalidOperationException("Forward3 must be called before Backward3.");
        var dLdx = new float[_lastX.Length][][];
        for (int b = 0; b < _lastX.Length; b++)
            dLdx[b] = MatrixHelper.LayerNormBackward(dLdOut[b], _lastX[b], _lastMean[b], _lastVar[b], _eps);
        return dLdx;
    }
}
