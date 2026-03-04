#if USE_TORCHSHARP_GPU
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using Embedding = TorchSharp.Modules.Embedding;

namespace TransformerDemo;

/// <summary>
/// 使用 TorchSharp + CUDA 在 GPU 上训练三国问答模型，训练完成后将权重导出为现有 C# 模型格式，供推理使用。
/// 需安装 TorchSharp 与 libtorch-cuda-12.1-win-x64（RTX 4070 等支持 CUDA 12 的显卡）。
/// </summary>
public static class SanguoGpuTrainer
{
    private const int PadId = 0;

    /// <summary>
    /// 在 GPU 上训练，并将权重导出到指定目录（与现有 C# 推理兼容）。
    /// </summary>
    public static void TrainOnGpu(
        IReadOnlyList<DataGenerator.Sample> trainData,
        IReadOnlyList<DataGenerator.Sample> validData,
        CharVocabulary vocab,
        int maxLen,
        int batchSize,
        int epochs,
        double learningRate,
        int dModel,
        int numHeads,
        int dFf,
        int numEncoderLayers,
        int numDecoderLayers,
        string outputDir)
    {
        if (!torch.cuda.is_available())
        {
            throw new InvalidOperationException("CUDA 不可用，请确保已安装 NVIDIA 驱动与 libtorch-cuda-12.1-win-x64（适配 RTX 4070 等）。");
        }

        var device = torch.CUDA;
        Console.WriteLine("使用 GPU: CUDA（如 RTX 4070）");

        int vocabSize = vocab.Size;

        // 与 C# 模型结构一致的 Transformer（便于导出权重）
        using var model = new Seq2SeqTransformer(
            vocabSize, dModel, numHeads, dFf, numEncoderLayers, numDecoderLayers, maxLen);
        model.to(device);

        // 使用 AdamW 更适合 Transformer，收敛更稳定（RTX 4070 上效果良好）
        var optimizer = torch.optim.AdamW(model.parameters(), learningRate, weight_decay: 0.01);

        int trainSteps = 0;
        foreach (int epoch in Enumerable.Range(1, epochs))
        {
            model.train();
            double totalLoss = 0;
            int batchCount = 0;
            foreach (var (encIds, decIds, validLengths) in DataGenerator.BatchesTokenized(trainData, maxLen, batchSize))
            {
                var enc = Int2LongTensor(encIds, device);
                var dec = Int2LongTensor(decIds, device);
                var validLen = validLengths;

                optimizer.zero_grad();
                var logits = model.forward(enc, dec, validLen, validLen, device);
                long[] flatTarget = FlattenTarget(decIds);
                var targetTensor = torch.tensor(flatTarget, device: device);
                var loss = torch.nn.functional.cross_entropy(logits, targetTensor, ignore_index: PadId);
                loss.backward();
                optimizer.step();

                totalLoss += loss.item<float>();
                batchCount++;
                trainSteps++;
            }
            double avgTrain = batchCount > 0 ? totalLoss / batchCount : 0;

            model.eval();
            double totalValid = 0;
            int validCount = 0;
            using (torch.no_grad())
            {
                foreach (var (encIds, decIds, validLengths) in DataGenerator.BatchesTokenized(validData, maxLen, batchSize))
                {
                    var enc = Int2LongTensor(encIds, device);
                    var dec = Int2LongTensor(decIds, device);
                    var logits = model.forward(enc, dec, validLengths, validLengths, device);
                    long[] flatTarget = FlattenTarget(decIds);
                    var targetTensor = torch.tensor(flatTarget, device: device);
                    var loss = torch.nn.functional.cross_entropy(logits, targetTensor, ignore_index: PadId);
                    totalValid += loss.item<float>();
                    validCount++;
                }
            }
            double avgValid = validCount > 0 ? totalValid / validCount : 0;
            Console.WriteLine($"Epoch {epoch,2}: Train Loss = {avgTrain:F4}, Valid Loss = {avgValid:F4}");
        }

        // 导出为 C# 模型格式
        ExportToCSharpFormat(model, vocabSize, dModel, numHeads, dFf, numEncoderLayers, numDecoderLayers, maxLen, outputDir, vocab);
        Console.WriteLine($"\nGPU 训练完成，模型已导出到: {Path.GetFullPath(outputDir)}");
    }

    private static long[] FlattenTarget(int[][] decIds)
    {
        var list = new List<long>();
        for (int b = 0; b < decIds.Length; b++)
            for (int s = 0; s < decIds[b].Length; s++)
                list.Add(decIds[b][s]);
        return list.ToArray();
    }

    private static torch.Tensor Int2LongTensor(int[][] ids, torch.Device device)
    {
        int batch = ids.Length, seq = ids[0].Length;
        var arr = new long[batch * seq];
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seq; s++)
                arr[b * seq + s] = ids[b][s];
        return torch.tensor(arr, device: device).view(batch, seq);
    }

    private static void ExportToCSharpFormat(
        Seq2SeqTransformer model,
        int vocabSize, int dModel, int numHeads, int dFf,
        int numEncoderLayers, int numDecoderLayers, int maxLen,
        string outputDir, CharVocabulary vocab)
    {
        Directory.CreateDirectory(outputDir);

        var csharpModel = new TransformerModel(vocabSize, dModel, numHeads, dFf, numEncoderLayers, numDecoderLayers, maxLen);
        var csharpParams = csharpModel.GetAllParameters();

        var torchParams = model.GetParametersInCSharpOrder();
        if (torchParams.Count != csharpParams.Count)
            throw new InvalidOperationException($"参数数量不一致: TorchSharp {torchParams.Count}, C# {csharpParams.Count}");

        for (int i = 0; i < csharpParams.Count; i++)
        {
            var csharp = csharpParams[i];
            var tensor = torchParams[i];
            int rows = csharp.Length, cols = csharp[0].Length;
            var data = tensor.cpu().data<float>().ToArray();
            // 第 0 个为 embedding (vocabSize, dModel)，与 PyTorch 同布局；其余为 Linear weight (out,in)，需按 (in,out) 转置写入 C#
            if (i == 0)
            {
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        csharp[r][c] = data[r * cols + c];
            }
            else
            {
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        csharp[r][c] = data[c * rows + r];
            }
        }

        csharpModel.SaveToDirectory(outputDir);
        vocab.SaveToDirectory(outputDir);
    }

    /// <summary>与 C# TransformerModel 结构一致的 Seq2Seq 模型，参数顺序与 GetAllParameters() 一致。</summary>
    private sealed class Seq2SeqTransformer : torch.nn.Module
    {
        private readonly int _vocabSize, _dModel, _numHeads, _dFf, _maxLen, _numEncLayers, _numDecLayers;
        private readonly TorchSharp.Modules.Embedding _embedding;
        private readonly torch.Tensor _pe;
        private readonly List<EncoderLayer> _encLayers;
        private readonly List<DecoderLayer> _decLayers;
        private readonly Linear _outputProj;

        public Seq2SeqTransformer(int vocabSize, int dModel, int numHeads, int dFf, int numEncLayers, int numDecLayers, int maxLen)
            : base(nameof(Seq2SeqTransformer))
        {
            _vocabSize = vocabSize;
            _dModel = dModel;
            _numHeads = numHeads;
            _dFf = dFf;
            _maxLen = maxLen;
            _numEncLayers = numEncLayers;
            _numDecLayers = numDecLayers;

            _embedding = torch.nn.Embedding(vocabSize, dModel);
            _pe = SinusoidalPE(maxLen, dModel);
            _encLayers = new List<EncoderLayer>();
            for (int i = 0; i < numEncLayers; i++)
            {
                var layer = new EncoderLayer(dModel, numHeads, dFf);
                register_module($"enc_{i}", layer);
                _encLayers.Add(layer);
            }
            _decLayers = new List<DecoderLayer>();
            for (int i = 0; i < numDecLayers; i++)
            {
                var layer = new DecoderLayer(dModel, numHeads, dFf);
                register_module($"dec_{i}", layer);
                _decLayers.Add(layer);
            }
            _outputProj = torch.nn.Linear(dModel, vocabSize);

            RegisterComponents();
            InitWeights();
        }

        private void InitWeights()
        {
            foreach (var p in parameters())
            {
                if (p.dim() < 2) continue;
                double bound = Math.Sqrt(6.0 / (p.shape[0] + p.shape[1]));
                torch.nn.init.uniform_(p, -bound, bound);
            }
        }

        private static torch.Tensor SinusoidalPE(int maxLen, int dModel)
        {
            var pe = torch.zeros(maxLen, dModel);
            var data = pe.data<float>();
            for (int pos = 0; pos < maxLen; pos++)
            {
                for (int i = 0; i < dModel; i++)
                {
                    double angle = pos / Math.Pow(10000.0, (2 * i) / (double)dModel);
                    float v = (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
                    data[pos,i] = v;
                }
            }
            return pe;
        }

        public torch.Tensor forward(torch.Tensor encIds, torch.Tensor decIds, int[] encValidLen, int[] decValidLen, torch.Device device)
        {
            int batch = (int)encIds.shape[0], encLen = (int)encIds.shape[1], decLen = (int)decIds.shape[1];
            var encEmb = _embedding.forward(encIds) + _pe.slice(0, 0, encLen, 1).to(device);
            var decEmb = _embedding.forward(decIds) + _pe.slice(0, 0, decLen, 1).to(device);

            var encMask = CreatePaddingMask(batch, encLen, encValidLen, device);
            var decMask = CreatePaddingMask(batch, decLen, decValidLen, device);
            var causalMask = CreateCausalMask(decLen, device);

            var encOut = encEmb;
            foreach (var layer in _encLayers)
                encOut = layer.forward(encOut, encMask);

            var decOut = decEmb;
            foreach (var layer in _decLayers)
                decOut = layer.forward(decOut, encOut, decMask, encMask, causalMask);

            var logits = _outputProj.forward(decOut);
            return logits.view(batch * decLen, _vocabSize);
        }

        private static torch.Tensor? CreatePaddingMask(int batch, int seqLen, int[] validLen, torch.Device device)
        {
            // key_padding_mask: True = 忽略该位置（padding）
            var mask = torch.zeros(batch, seqLen, dtype: torch.ScalarType.Bool).to(device);
            for (int b = 0; b < batch; b++)
                for (int s = validLen[b]; s < seqLen; s++)
                    mask[b, s] = true;
            return mask;
        }

        private static torch.Tensor CreateCausalMask(int seqLen, torch.Device device)
        {
            var mask = torch.tril(torch.ones(seqLen, seqLen, dtype: torch.ScalarType.Bool)).to(device);
            return mask;
        }

        /// <summary>返回与 C# GetAllParameters() 顺序一致的参数列表（仅权重矩阵，用于导出）。</summary>
        public List<torch.Tensor> GetParametersInCSharpOrder()
        {
            var list = new List<torch.Tensor>();
            list.Add(_embedding.weight);
            list.Add(_outputProj.weight);
            foreach (var layer in _encLayers)
                layer.CollectParameters(list);
            foreach (var layer in _decLayers)
                layer.CollectParameters(list);
            return list;
        }
    }

    private sealed class EncoderLayer : torch.nn.Module
    {
        private readonly MultiHeadAttentionLayer _selfAttn;
        private readonly SimpleLayerNorm _norm1, _norm2;
        private readonly Linear _ffn1, _ffn2;

        public EncoderLayer(int dModel, int numHeads, int dFf)
            : base("Enc")
        {
            _selfAttn = new MultiHeadAttentionLayer(dModel, numHeads);
            _norm1 = new SimpleLayerNorm(dModel);
            _ffn1 = torch.nn.Linear(dModel, dFf);
            _ffn2 = torch.nn.Linear(dFf, dModel);
            _norm2 = new SimpleLayerNorm(dModel);
            RegisterComponents();
        }

        public torch.Tensor forward(torch.Tensor x, torch.Tensor? paddingMask)
        {
            var attnOut = _selfAttn.forward(x, x, x, paddingMask, null);
            x = x + attnOut;
            x = _norm1.forward(x);
            var ffnOut = torch.nn.functional.relu(_ffn1.forward(x));
            ffnOut = _ffn2.forward(ffnOut);
            x = x + ffnOut;
            x = _norm2.forward(x);
            return x;
        }

        public void CollectParameters(List<torch.Tensor> list)
        {
            _selfAttn.CollectParameters(list);
            list.Add(_ffn1.weight);
            list.Add(_ffn2.weight);
        }
    }

    private sealed class DecoderLayer : torch.nn.Module
    {
        private readonly MultiHeadAttentionLayer _selfAttn, _crossAttn;
        private readonly SimpleLayerNorm _norm1, _norm2, _norm3;
        private readonly Linear _ffn1, _ffn2;

        public DecoderLayer(int dModel, int numHeads, int dFf)
            : base("Dec")
        {
            _selfAttn = new MultiHeadAttentionLayer(dModel, numHeads);
            _norm1 = new SimpleLayerNorm(dModel);
            _crossAttn = new MultiHeadAttentionLayer(dModel, numHeads);
            _norm2 = new SimpleLayerNorm(dModel);
            _ffn1 = torch.nn.Linear(dModel, dFf);
            _ffn2 = torch.nn.Linear(dFf, dModel);
            _norm3 = new SimpleLayerNorm(dModel);
            RegisterComponents();
        }

        public torch.Tensor forward(torch.Tensor decInput, torch.Tensor encOutput, torch.Tensor? decPadMask, torch.Tensor? encPadMask, torch.Tensor causalMask)
        {
            var selfOut = _selfAttn.forward(decInput, decInput, decInput, decPadMask, causalMask);
            decInput = decInput + selfOut;
            decInput = _norm1.forward(decInput);
            var crossOut = _crossAttn.forward(decInput, encOutput, encOutput, encPadMask, null);
            decInput = decInput + crossOut;
            decInput = _norm2.forward(decInput);
            var ffnOut = torch.nn.functional.relu(_ffn1.forward(decInput));
            ffnOut = _ffn2.forward(ffnOut);
            decInput = decInput + ffnOut;
            return _norm3.forward(decInput);
        }

        public void CollectParameters(List<torch.Tensor> list)
        {
            _selfAttn.CollectParameters(list);
            _crossAttn.CollectParameters(list);
            list.Add(_ffn1.weight);
            list.Add(_ffn2.weight);
        }
    }

    /// <summary>无参数 LayerNorm：对最后一维做 (x-mean)/sqrt(var+eps)，与 C# 行为一致。</summary>
    private sealed class SimpleLayerNorm : torch.nn.Module
    {
        private readonly int _dModel;
        private const float Eps = 1e-6f;

        public SimpleLayerNorm(int dModel) : base("LayerNorm") { _dModel = dModel; }

        public torch.Tensor forward(torch.Tensor x)
        {
            var mean = x.mean(new long[] { -1 }, true);
            var var = ((x - mean).pow(2)).mean(new long[] { -1 }, true);
            return (x - mean) / (var + Eps).sqrt();
        }
    }

    /// <summary>多头注意力，用 4 个 Linear 表示 Q,K,V,O，与 C# 参数顺序一致。</summary>
    private sealed class MultiHeadAttentionLayer : torch.nn.Module
    {
        private readonly int _dModel, _numHeads, _dK;
        private readonly Linear _wQ, _wK, _wV, _wO;

        public MultiHeadAttentionLayer(int dModel, int numHeads)
            : base("MHA")
        {
            _dModel = dModel;
            _numHeads = numHeads;
            _dK = dModel / numHeads;
            _wQ = torch.nn.Linear(dModel, dModel);
            _wK = torch.nn.Linear(dModel, dModel);
            _wV = torch.nn.Linear(dModel, dModel);
            _wO = torch.nn.Linear(dModel, dModel);
            RegisterComponents();
        }

        public torch.Tensor forward(torch.Tensor q, torch.Tensor k, torch.Tensor v, torch.Tensor? keyPaddingMask, torch.Tensor? attnMask)
        {
            int batch = (int)q.shape[0], seqQ = (int)q.shape[1], seqK = (int)k.shape[1];
            var qProj = _wQ.forward(q).view(batch, seqQ, _numHeads, _dK).transpose(1, 2);
            var kProj = _wK.forward(k).view(batch, seqK, _numHeads, _dK).transpose(1, 2);
            var vProj = _wV.forward(v).view(batch, seqK, _numHeads, _dK).transpose(1, 2);

            double scale = 1.0 / Math.Sqrt(_dK);
            var scores = torch.matmul(qProj, kProj.transpose(-2, -1)) * scale;
            if (attnMask is not null)
                scores = scores.masked_fill(attnMask.unsqueeze(0).unsqueeze(0) == false, float.NegativeInfinity);
            if (keyPaddingMask is not null)
            {
                var km = keyPaddingMask.unsqueeze(1).unsqueeze(2);
                scores = scores.masked_fill(km, float.NegativeInfinity);
            }
            var attn = torch.nn.functional.softmax(scores, dim: -1);
            var out_ = torch.matmul(attn, vProj).transpose(1, 2).contiguous().view(batch, seqQ, _dModel);
            return _wO.forward(out_);
        }

        public void CollectParameters(List<torch.Tensor> list)
        {
            list.Add(_wQ.weight);
            list.Add(_wK.weight);
            list.Add(_wV.weight);
            list.Add(_wO.weight);
        }
    }
}
#endif
