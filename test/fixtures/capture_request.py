"""
从 DataHub 侧导出真实请求体到本地文件。

用法：
  # 在 DataHub 执行时，通过 shell 管道或日志导出一份 JSON 到本目录
  # 例如 DataHub 发送请求前先写入文件：
  python -c "
    import json
    payload = { ... }  # DataHub 拼装好的请求体
    with open('test/fixtures/request_real.json', 'w') as f:
      json.dump(payload, f, indent=2)
  "
"""

# 这里不需要实际代码，仅作说明。
# 实际捕获由 DataHub 侧完成，MVP 侧负责接收和回放。
