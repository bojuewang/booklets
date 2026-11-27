# Booklet 编辑规范

## 目录结构

- 文件与目录命名遵循「工程词典」的命名规范。[Engineering Dictionary](https://github.com/bojuewang/engineering-dictionary)
- 新增一册时使用 `<topic>/booklet.md` 作为正文入口，首次加入插图时再创建 `<topic>/figures/`。
- 主题目录和图文件名使用小写英文，单词之间使用单个连字符。
- 新增一册时同步更新根目录 `README.md` 的总索引，直接链接该册的正文入口并标注内容状态。

## 术语与关键词

- 英文术语、关键词、缩写和字段名使用 Markdown 行内代码格式，例如 `distance correlation`、`dcor` 和 `rank_ic`。
- 中文术语和关键词使用直角引号，例如「距离相关」和「参数估计」。
- 中英文并列时，写作「距离相关」（`distance correlation`，`dcor`）。
- 数学公式沿用 LaTeX 记号，例如 `$\operatorname{dcor}(U,V)$`。
