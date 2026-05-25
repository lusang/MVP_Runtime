# 测试夹具（Test Fixtures）

本目录存放从 DataHub 导出的真实请求 JSON，用于 MVP 本地测试。

## 用法

1. 从 DataHub 侧获取真实请求体（见下方 `capture_request.py`）
2. 保存为 `test/fixtures/request_{描述}.json`
3. MVP 侧测试脚本加载此文件，直接 `POST /run_annotation_async`

## 如何从 DataHub 捕获真实请求

### 方法 1：DataHub 日志导出

在 DataHub 调用 `POST /run_annotation_async` 的地方，将请求体
打印到日志或导出到文件，格式为：

```json
{
  "template": { ... },
  "callback_url": "http://mvp:8001/run_annotation_async",
  "tasks": [ ... ]
}
```

### 方法 2：使用 capture_request.py

在 DataHub 侧运行此脚本（需修改目标 URL），
它将接收一个完整的 MVP 请求体并保存到本地文件。

```bash
# 在 DataHub 侧执行
python capture_request.py --output request_real.json
```
