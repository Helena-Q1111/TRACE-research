# Sheet Sage - HookTheory Dataset

该数据集来源于 [Sheet Sage](https://github.com/chrisdonahue/sheetsage) 项目，包含约50小时的对齐旋律和和声标注数据，源自 [Hooktheory's TheoryTab DB](https://www.hooktheory.com/theorytab)。

**许可证**: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US) (非商业用途)

## 文件说明

| 文件 | 说明 | 大小 |
|------|------|------|
| `Hooktheory.json.gz` | **主数据集** - MIR友好的JSON格式完整数据集 | ~20MB |
| `Hooktheory_Raw.json.gz` | 原始HookTheory私有格式的完整数据集 | ~90MB |
| `Hooktheory_Test_Segments.json` | 测试集输入（用于官方评估） | ~200KB |
| `Hooktheory_Test_MIDI.tar.gz` | 测试集参考输出MIDI（用于官方评估） | ~300KB |
| `Hooktheory_Train_Segments.json` | 训练集输入 | ~1.8MB |
| `Hooktheory_Train_MIDI.tar.gz` | 训练集MIDI参考 | ~2.6MB |
| `Hooktheory_Valid_Segments.json` | 验证集输入 | ~180KB |
| `Hooktheory_Valid_MIDI.tar.gz` | 验证集MIDI参考 | ~260KB |

## 数据集结构

数据按 artist name 分层划分为 train:validation:test = 8:1:1。

### JSON格式 (Hooktheory.json.gz)

解压后为JSON对象，每个标注以HookTheory ID为键：

```python
import gzip
import json

with gzip.open('Hooktheory.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

# 查看一条数据
for key, annotation in list(data.items())[:1]:
    print(f"ID: {key}")
    print(f"Split: {annotation['split']}")  # train/valid/test
    print(f"Tags: {annotation['tags']}")
```

### 重要字段

- `split`: 数据划分 (train/valid/test)
- `tags`: 高级标签
- `alignment`: 包含原始对齐和精细对齐

### 推荐的数据过滤（训练旋律转录模型）

筛选包含以下标签的数据:
- ✅ `AUDIO_AVAILABLE` - 有音频可用
- ✅ `MELODY` - 包含旋律

排除包含以下标签的数据:
- ❌ `TEMPO_CHANGES` - 包含tempo变化

```python
def filter_for_melody_transcription(data):
    filtered = {}
    for key, ann in data.items():
        tags = ann.get('tags', [])
        if 'AUDIO_AVAILABLE' in tags and 'MELODY' in tags and 'TEMPO_CHANGES' not in tags:
            filtered[key] = ann
    return filtered
```

## 解压 tar.gz 文件

```bash
# Linux/Mac
tar -xzf Hooktheory_Test_MIDI.tar.gz
tar -xzf Hooktheory_Train_MIDI.tar.gz
tar -xzf Hooktheory_Valid_MIDI.tar.gz

# Windows PowerShell
tar -xzf Hooktheory_Test_MIDI.tar.gz
# 或使用 7-Zip
```

## 评估新系统

评估旋律转录系统时：

1. **输入**: 使用 `Hooktheory_Test_Segments.json` 中的信息（音频标识符和可选的片段边界）
2. **输出**: 让系统输出MIDI文件（以数据集key命名）到指定目录
3. **评估**: 运行以下命令：

```bash
python -m sheetsage.eval Hooktheory_Test_MIDI.tar.gz <MY_OUTPUT_DIR> --allow_abstain
```

## Sheet Sage 使用方法

### 快速开始（Linux + Docker）

```bash
# 一次性设置（下载~4GB Docker容器和~100MB数据）
ROOT=https://raw.githubusercontent.com/chrisdonahue/sheetsage/main
wget $ROOT/prepare.sh && wget $ROOT/sheetsage.sh && chmod +x *.sh && ./prepare.sh

# 转录YouTube视频
./sheetsage.sh https://www.youtube.com/watch?v=fHI8X4OXluQ

# 转录本地文件
./sheetsage.sh my_song.mp3
```

输出目录 `output/<UUID>` 包含:
- PDF乐谱（带LilyPond源文件）
- MIDI文件（包含音频对齐的旋律和和声）

### 改进转录效果

| 问题 | 解决方案 |
|------|----------|
| 只转录短片段（~24秒） | `./sheetsage.sh -s 17 --legacy_behavior <song>` |
| 速度检测错误（半速/倍速） | `./sheetsage.sh --beats_per_minute_hint 120 <song>` |
| 下拍检测错误 | `./sheetsage.sh -s 17.0 -e 40.0 --segment_hints_are_downbeats <song>` |
| 使用Jukebox（需GPU≥12GB） | `./prepare.sh -j` 然后 `./sheetsage.sh -j <song>` |

## 引用

```bibtex
@inproceedings{donahue2022melody,
  title={Melody transcription via generative pre-training},
  author={Donahue, Chris and Thickstun, John and Liang, Percy},
  booktitle={ISMIR},
  year={2022}
}
```

## 相关链接

- [Sheet Sage 论文](https://arxiv.org/abs/2212.01884)
- [GitHub 仓库](https://github.com/chrisdonahue/sheetsage)
- [数据集探索 Notebook](https://github.com/chrisdonahue/sheetsage/blob/main/notebooks/Dataset.ipynb)
- [HookTheory TheoryTab](https://www.hooktheory.com/theorytab)
