# Anima 28→40 批量 LoRA 转换器

[English](#anima-2840-batch-lora-converter) | 简体中文

把一批 **28 层 Anima LoRA** 的键名重映射为 **40 层** 结构，并批量保存到 ComfyUI 的 `models/loras` 文件夹。参照 [ComfyUI-Anima-28to40-Lora-Stack](https://github.com/hpoc766-afk/ComfyUI-Anima-28to40-Lora-Stack) 的映射规范（插入层位置 2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36），与内存版堆叠节点不同，本节点是**离线批量转换器**：直接写出转换后的 `.safetensors` 文件。

## 安装

最简单的方式是在 ComfyUI-Manager 里搜索 `anima-28to40-lora-converter` 一键安装。

也可以手动克隆：

把仓库克隆到 `ComfyUI/custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/R0smontis/ComfyUI-Anima-28to40-Lora-Converter.git
```

重启 ComfyUI。无额外 Python 依赖。

## 用法

节点：`loaders/Anima > Anima 28→40 批量 LoRA 转换器`（输出节点，直接排队执行）。

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `source_mode` | `names` | `names` 严格模式：逐行填写 loras 相对路径（换行/逗号/分号分隔）；`folder` 宽容模式：递归扫描子文件夹 |
| `lora_names` | 空 | names 模式的文件列表；点「＋ 浏览添加 LoRA…」可搜索多选 |
| `source_subfolder` | `anima` | folder 模式的扫描子文件夹（相对 `models/loras`） |
| `filter` | `*` | folder 模式文件名过滤：`*`/`?` 通配符，或无通配符时按子串匹配（不区分大小写） |
| `output_subfolder` | `anima_40` | 输出子文件夹（相对 `models/loras`） |
| `name_suffix` | `_40` | 追加到文件名末尾（已带后缀则不重复追加） |
| `overwrite` | `false` | 目标已存在时：`false` 跳过并记入摘要；`true` 覆盖 |

### 行为

- **键名前缀**：支持 `lora_unet_blocks_*`、`net.blocks.*`、`diffusion_model.blocks.*` 三种命名的 28 层 LoRA（映射如 `0→0`、`2→3`、`14→20`、`27→39`）。
- **输出路径**：`models/loras/<output_subfolder>/` 下保留源文件相对子目录；names 模式保留填写路径的子目录（如 `anima/foo.safetensors` → `anima_40/anima/foo_40.safetensors`），folder 模式剥掉扫描根前缀。
- **元数据**：原样保留并追加 `ss_anima_remap = "28to40"` 标记。
- **严格校验**：无主干层键、索引超出 0–27、映射后键名冲突、输出与源同路径、不同源指向同一目标，均在写文件前整体报错；names 模式中途失败立即中止并说明已写出数量，folder 模式逐文件跳过并记入摘要。
- 已转换过的 40 层文件再次被选中时会报「该文件可能已经是 40 层结构」，不会被二次转换。

## 开发与测试

```bash
python -m unittest discover -s tests -v
node --check web/anima_lora_converter.js
```

## License

[MIT](LICENSE)

---

# Anima 28→40 Batch LoRA Converter

Offline batch converter that remaps **28-layer Anima LoRA** keys to the **40-layer** structure and saves the converted `.safetensors` files into ComfyUI's `models/loras` folder. Same mapping spec as [ComfyUI-Anima-28to40-Lora-Stack](https://github.com/hpoc766-afk/ComfyUI-Anima-28to40-Lora-Stack) (insertion layers at 2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36), but writes converted files instead of patching in memory.

## Install

Easiest: search `anima-28to40-lora-converter` in ComfyUI-Manager. Or clone manually:


```bash
cd ComfyUI/custom_nodes
git clone https://github.com/R0smontis/ComfyUI-Anima-28to40-Lora-Converter.git
```

Restart ComfyUI. No extra Python dependencies.

## Usage

Node: `loaders/Anima > Anima 28→40 批量 LoRA 转换器` (output node; queue to run).

- `names` mode (strict): list loras relative to `models/loras`, one per line (newline/comma/semicolon separated). Any error aborts before writing.
- `folder` mode (lenient): recursively scan `source_subfolder`, filter file names, un-convertible files are skipped and listed in the summary.
- Outputs go to `models/loras/<output_subfolder>` (default `anima_40`), keeping the source subfolder structure, with `name_suffix` appended (default `_40`). Metadata is preserved plus a `ss_anima_remap = "28to40"` marker.
- Supports `lora_unet_blocks_*`, `net.blocks.*` and `diffusion_model.blocks.*` key prefixes.

## Tests

```bash
python -m unittest discover -s tests -v
node --check web/anima_lora_converter.js
```

## License

[MIT](LICENSE)
