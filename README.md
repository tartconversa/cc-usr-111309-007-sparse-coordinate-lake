# 稀疏实验张量的坐标分块目录

本项目保存虚构实验设施的仓湖源数据与Python读取工具。现有工具只加载文件并显示条数，不执行事实加工、解码、合并、联接或重映射。全部设备、样品、记录和数值均为合成。

## 字段与单位

coordinate按tensor中的轴顺序解释，value可为0或null，tombstone是真删除；未出现坐标不是一条null记录。块索引与重分块尚未实现。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 运行已有读取器

Python 3.11以上，标准库即可，不需安装第三方依赖。

    python -m src.coordinate_reader fixtures/coordinates.jsonl
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

## 固定容器

    docker build -t sparse-coordinate-lake:local .
    docker run --rm sparse-coordinate-lake:local

容器镜像以完整digest固定，默认仅检查输入契约；实现新增分析功能所需依赖由开发者明确登记。
