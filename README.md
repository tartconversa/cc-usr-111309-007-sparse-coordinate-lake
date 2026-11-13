# 稀疏实验张量的坐标分块目录

本项目保存虚构实验设施的仓湖源数据与Python读取工具。现有工具只加载文件并显示条数，不执行事实加工、解码、合并、联接或重映射。全部设备、样品、记录和数值均为合成。

## 字段与单位

coordinate按tensor中的轴顺序解释，value可为0或null，tombstone是真删除；未出现坐标不是一条null记录。

每快照固定轴顺序、各维长度与chunk_shape；坐标必须落在`[0,length)`。逻辑记录含完整坐标、数值或显式null、revision与tombstone。同一坐标按最高revision裁决，最高revision平级冲突拒绝，tombstone移除坐标。三种状态为：

- `absent`：没有记录
- `present_null`：有记录但值为null
- `present_value`：有记录且为数值，0与负数都是真实数值

块键为逐维`coordinate // chunk_shape`，块内位置为逐维`coordinate % chunk_shape`，再以该块的有效边缘形状（最后一块可小于完整尺寸）按行主序展平为offset。目录只保存非空块，越界padding不算缺测，也不会为稀疏查询先建完整网格。

稀疏实现位于`src/sparse_lake.py`：

    python -m src.sparse_lake fixtures/coordinates.jsonl fixtures/tensor.json \
        --axes device tick --window 0 0 2 5 --rechunk 1 2

窗口参数按各维先lower后upper；`--dense`时才打印密集导出。

`Snapshot`构建不可变块、坐标索引与清单，支持`restore`按坐标恢复、`scan_window`矩形半开窗口扫描、`window_plan`逐块交集与有效边缘形状、`rechunk`重分块。重分块只改变物理分块，逻辑坐标到状态映射以`logical_digest`核对，与按sum比较不同。

`dense_export`仅在显式调用时生成密集表示，并同时给出`presence`与`validity`位图：`presence=1,validity=0`是显式null，`presence=0`是未记录，缺失不会自动填0。

shape=(3,5)、chunk=(2,3)时，坐标(2,4)的块键是(1,1)，块内位置是(0,1)，该边缘块有效shape=(1,2)，行主序offset=1。fixtures中(0,0)的值0、(0,1)的null与(0,2)的未记录可通过`restore`或位图区分。

## 目录与边界

目录以坐标称为单位，不拟合物理张量、不运行量子算法，也不把实验缺口当作可插值的物理量。块编码采用行主序稀疏offset表，即使大稀疏域也只有非空块与已记录条目占内存；`footprint`给出逻辑格数、已记录数、空块数与块网格规模，便于说明未建密集数组。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 运行已有读取器

Python 3.11以上，标准库即可，不需安装第三方依赖。

    python -m src.coordinate_reader fixtures/coordinates.jsonl
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

## 固定容器

    docker build -t sparse-coordinate-lake:local .
    docker run --rm sparse-coordinate-lake:local

容器镜像以完整digest固定，默认仅检查输入契约；本次稀疏湖表组件只使用标准库，未新增依赖。
