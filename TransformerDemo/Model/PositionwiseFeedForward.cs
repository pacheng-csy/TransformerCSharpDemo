using System;

namespace TransformerDemo;

/// <summary>
/// 位置前馈网络（Position-wise Feed-Forward）
/// 两层线性变换 + 中间 ReLU：FFN(x) = W2 * ReLU(W1 * x)。
/// 每个位置独立计算，等价于在序列长度维上做 1x1 卷积（即逐位置的全连接）。
/// 维度：d_model -> d_ff -> d_model。
/// </summary>
public class PositionwiseFeedForward
{
    private readonly int _dModel, _dFf;
    private readonly float[][] _w1, _w2;
    private readonly float[][] _gradW1, _gradW2;
    private float[][][]? _lastX;
    private float[][][]? _lastPreRelu;  // W1*x，ReLU 前

    public PositionwiseFeedForward(int dModel, int dFf)
    {
        _dModel = dModel;
        _dFf = dFf;
        _w1 = MatrixHelper.Zeros(dModel, _dFf);
        _w2 = MatrixHelper.Zeros(_dFf, dModel);
        _gradW1 = MatrixHelper.Zeros(dModel, _dFf);
        _gradW2 = MatrixHelper.Zeros(_dFf, dModel);
        MatrixHelper.XavierUniform(_w1);
        MatrixHelper.XavierUniform(_w2);
    }

    public void ZeroGrad()
    {
        for (int i = 0; i < _gradW1.Length; i++)
            for (int j = 0; j < _gradW1[i].Length; j++)
                _gradW1[i][j] = 0;
        for (int i = 0; i < _gradW2.Length; i++)
            for (int j = 0; j < _gradW2[i].Length; j++)
                _gradW2[i][j] = 0;
    }

    public float[][][] Forward(float[][][] x)
    {
        var preRelu = MatrixHelper.MultiplyBatch3D(x, _w1);
        _lastX = MatrixHelper.Copy3(x);
        _lastPreRelu = MatrixHelper.Copy3(preRelu);
        for (int b = 0; b < preRelu.Length; b++)
            preRelu[b] = MatrixHelper.ReLU(preRelu[b]);
        return MatrixHelper.MultiplyBatch3D(preRelu, _w2);
    }

    /// <summary>反向：dLdOut 与输出同形状，返回 dLdX。</summary>
    public float[][][] Backward(float[][][] dLdOut)
    {
        if (_lastX == null || _lastPreRelu == null)
            throw new InvalidOperationException("Forward must be called before Backward.");
        var reluOut = MatrixHelper.Copy3(_lastPreRelu);
        for (int b = 0; b < reluOut.Length; b++)
            reluOut[b] = MatrixHelper.ReLU(reluOut[b]);
        var dLdReluOut = MatrixHelper.MultiplyBatch3DBackward(dLdOut, reluOut, _w2, _gradW2);
        var dLdPreRelu = new float[_lastPreRelu.Length][][];
        for (int b = 0; b < dLdPreRelu.Length; b++)
            dLdPreRelu[b] = MatrixHelper.ReLUBackward(dLdReluOut[b], _lastPreRelu[b]);
        return MatrixHelper.MultiplyBatch3DBackward(dLdPreRelu, _lastX, _w1, _gradW1);
    }

    public (float[][] w1, float[][] w2) GetParameters() => (_w1, _w2);
    public (float[][] gradW1, float[][] gradW2) GetGradients() => (_gradW1, _gradW2);
}
