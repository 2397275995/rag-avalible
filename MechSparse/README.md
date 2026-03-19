### MechSparse（电路引导 SAE + 条件互信息 + GAM + AARF+）复现实验指南

本目录把你论文里描述的 **MechSparse** 方案落成到当前工作区代码（`ReDEeP-ICLR` + `RAGLens`）的可运行管线。

> 说明：由于你仓库里 ReDeEP 代码原始写法包含 Linux 绝对路径（模型目录、BGE 模型目录），本指南将关键路径全部改为**命令行参数/环境变量**方式。你只需要把 `--model_name_or_path` 和 SAE 路径设置到你本机即可。

---

### 1) 产物约定（你写论文时也建议按这个引用）

- **电路与数据集（ReDeEP 侧导出）**
  - `mechsparse_circuits.json`：复制头集合 𝒜（layer/head）与知识 FFN 层集合 𝓕（layer id）
  - `mechsparse_redeep_chunk_dataset.jsonl`：每个 chunk/span 一行，包含
    - `input` / `output`
    - `label`（0/1）
    - `H`（ReDeEP 风格代理分数）

- **检测器（RAGLens 侧训练后保存）**
  - `mechsparse_detector.pkl`：EBM/GAM + top 特征索引 +（可选）电路参数（不包含 tokenizer/model/SAE）

---

### 2) 第一步：准备 ReDeEP chunk-level 输出

你需要先跑 `ReDEeP-ICLR/ReDeEP/chunk_level_detect.py` 得到形如：

- `ReDEeP-ICLR/ReDeEP/log/<exp_name>/llama*_response_chunk.json`

> 这个 JSON 需要包含：`source_id`, `response`, `response_spans`, `scores[*].prompt_attention_score`, `scores[*].parameter_knowledge_scores`, `scores[*].hallucination_label`。

---

### 3) 第二步：导出 MechSparse 电路（𝒜/𝓕）与训练 JSONL（含 H）

在 `ReDEeP-ICLR/ReDeEP` 目录运行：

```bash
python mechsparse_extract.py ^
  --response_chunk_json "<你的 llm*_response_chunk.json 路径>" ^
  --source_info_jsonl "<ReDEeP 的 source_info*.jsonl 路径>" ^
  --splits "train,test" ^
  --top_n 32 ^
  --pcc_ext_threshold -0.4 ^
  --pcc_par_threshold 0.3 ^
  --late_layer_min 18 ^
  --alpha 0.6 ^
  --m 1.0 ^
  --out_dir "<输出目录>"
```

输出目录将包含：
- `mechsparse_circuits.json`
- `mechsparse_redeep_chunk_dataset.jsonl`

---

### 4) 第三步：训练 MechSparse Detector（SAE + 条件互信息 + GAM）

先准备一个 **SAE checkpoint**（`torch.load()` 可读）。SAE 的输入维度应为 `2 * hidden_size`，因为 MechSparse 编码使用 `[r_ext; r_par]` 拼接。

然后运行：

```bash
python RAGLens/src/run_mechsparse_train_detector.py ^
  --model_name_or_path "<本机HF模型路径或模型名>" ^
  --sae_path "<你的 SAE checkpoint>" ^
  --dataset_jsonl "<上一步的 mechsparse_redeep_chunk_dataset.jsonl>" ^
  --circuits_json "<上一步的 mechsparse_circuits.json>" ^
  --top_k 64 ^
  --out_pkl "<输出 mechsparse_detector.pkl>"
```

训练逻辑：
- 编码：`encode_mechsparse_outputs()`（只看 copy-head 与 knowledge-FFN 的残差贡献）
- 特征选择：`I(z;label|H)`（`utils.compute_conditional_mutual_information_chunked`）
- 模型：EBM/GAM（`interpret.glassbox.ExplainableBoostingClassifier`）
- 线性协变量：把 `H` 拼到特征最后一列

---

### 5) 第四步：AARF+（生成时干预，特征感知重加权）

`ReDEeP-ICLR/AARF/mechsparse_aarf_plus.py` 提供一个可运行原型：

```bash
set MECHSPARSE_SAE_PATH=<你的 SAE checkpoint>
python ReDEeP-ICLR/AARF/mechsparse_aarf_plus.py ^
  --model_name_or_path "<本机HF模型路径或模型名>" ^
  --detector_pkl "<mechsparse_detector.pkl>" ^
  --prompt "<你的测试prompt>" ^
  --max_new_tokens 128 ^
  --temperature 0.7 ^
  --tau 0.5 ^
  --alpha2 1.3 ^
  --beta2 0.7
```

实现细节（与论文一致、但工程上做了最小可跑近似）：
- 当检测器预测风险 `p_hall > tau`：
  - 放大 copy circuit 所在层的 attention 输出（layer-level）× `alpha2`
  - 衰减 knowledge FFN 层的 down_proj 输出（layer-level）× `beta2`

---

### 6) 论文可复用的实验段落建议

- **检测（RAGTruth/Dolly/AggreFact/TofuEval）**：报告 AUC / F1 / Acc，并加消融：
  - 去电路引导（改回 `encode_outputs`）
  - 去条件互信息（改回 `compute_mutual_information_chunked`）
  - 去 H 线性项（训练/预测时不传 `H_values`）

- **缓解（AARF vs AARF+）**：报告 hallucination rate，建议同时给：
  - 触发率（多少 token/step 触发了干预）
  - 触发后成功转化率（高风险 → 低风险）

