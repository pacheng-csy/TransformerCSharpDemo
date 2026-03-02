using System;

namespace TransformerDemo;

/// <summary>
/// 词嵌入层：将 token id 映射为 d_model 维向量。查表实现。
/// 支持反向传播：Backward 将 dLdOut 按 ids 累加到 _gradTable。
/// </summary>
public class Embedding
{
    private readonly float[][] _table;   // [vocabSize, dModel]
    private readonly float[][] _gradTable;

    public Embedding(int vocabSize, int dModel)
    {
        _table = MatrixHelper.Zeros(vocabSize, dModel);
        _gradTable = MatrixHelper.Zeros(vocabSize, dModel);
        MatrixHelper.XavierUniform(_table);
    }

    /// <summary>ids: (batch, seqLen)，返回 (batch, seqLen, d_model)</summary>
    public float[][][] Forward(int[][] ids)
    {
        int batch = ids.Length, seqLen = ids[0].Length, dModel = _table[0].Length;
        var out_ = MatrixHelper.Zeros3(batch, seqLen, dModel);
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int id = ids[b][s];
                if (id < 0 || id >= _table.Length) continue;
                for (int d = 0; d < dModel; d++)
                    out_[b][s][d] = _table[id][d];
            }
        return out_;
    }

    /// <summary>将梯度清零，每次训练步前调用。</summary>
    public void ZeroGrad()
    {
        for (int i = 0; i < _gradTable.Length; i++)
            for (int j = 0; j < _gradTable[i].Length; j++)
                _gradTable[i][j] = 0;
    }

    /// <summary>反向传播：dLdOut (batch, seqLen, dModel)，按 ids 累加到 _gradTable。</summary>
    public void Backward(float[][][] dLdOut, int[][] ids)
    {
        MatrixHelper.ScatterAddIntoRows(_gradTable, ids, dLdOut);
    }

    public float[][] GetTable() => _table;
    public float[][] GetGradTable() => _gradTable;
}
