# MVP Runtime — DataHub 集成接口文档

> 版本：v1.1  
> 更新日期：2026-05-23  
> 状态：部分已实现（健康检查 ✅）

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DataHub                                    │
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────────────┐   │
│  │   数据库   │───→│  业务逻辑    │───→│ POST /run_annotation_async│   │
│  │ (模板JSON) │    │ (组装请求)   │    │  → MVP Runtime            │   │
│  └──────────┘    └─────────────┘    └───────────┬───────────────┘   │
│                                                  │                   │
│  ┌──────────────────────────────────────┐        │                   │
│  │ ← POST /api/annotation_callback      │◄───────┘                   │
│  │  (MVP 处理完主动回调)                  │                           │
│  └──────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MVP Runtime                                   │
│                                                                     │
│  1. 接收请求 → 解析 template + tasks                                │
│  2. 解析 URL (file:// 直接读, http:// 下载到临时目录)                │
│  3. 对每个 task 逐帧执行推理                                         │
│  4. 每完成一个 task → POST 回调到 callback_url                      │
│  5. 清理临时文件                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 接口定义

### 2.1 提交批处理任务

```
POST /run_annotation_async
Content-Type: application/json
Accept: application/json
```

#### 请求体

```json
{
  "template": {
    "objects": [
      {
        "name": "包裹",
        "description": "快递配送场景中的包裹物体",
        "include": "包裹, 快递盒, 纸箱, 快递袋",
        "exclude": "人, 车辆, 宠物, 植物",
        "attributes": [
          {
            "name": "颜色",
            "type": "multi_select",
            "options": ["红色", "蓝色", "白色", "黑色"],
            "description": "包裹主体颜色"
          },
          {
            "name": "有快递单",
            "type": "boolean",
            "options": [],
            "description": "包裹表面是否有快递运单/标签"
          }
        ],
        "quality": {
          "attributes": [
            {
              "name": "blur",
              "type": "select",
              "options": ["清晰", "轻微模糊", "重度模糊"],
              "description": "图像模糊程度"
            },
            {
              "name": "lighting",
              "type": "select",
              "options": ["正常", "偏暗", "过曝"],
              "description": "光照条件"
            },
            {
              "name": "occlusion",
              "type": "select",
              "options": ["无遮挡", "部分遮挡", "严重遮挡"],
              "description": "遮挡程度"
            }
          ]
        },
        "negative": {
          "attributes": [
            {
              "name": "纯负样本",
              "type": "boolean",
              "options": [],
              "description": "场景中完全没有目标物体"
            }
          ]
        }
      }
    ]
  },
  "callback_url": "http://datahub:8000/api/annotation_callback",
  "tasks": [
    {
      "task_id": "batch_20260523_001",
      "media_type": "image",
      "frames": [
        {
          "frame_id": "img_001",
          "url": "file:///D:/datahub/images/img_001.jpg",
          "timestamp_ms": 0
        }
      ]
    },
    {
      "task_id": "clip_20260523_002",
      "media_type": "video_clip",
      "frames": [
        { "frame_id": "f001", "url": "file:///D:/datahub/video/frame_0001.jpg", "timestamp_ms": 0 },
        { "frame_id": "f002", "url": "file:///D:/datahub/video/frame_0002.jpg", "timestamp_ms": 33 },
        { "frame_id": "f003", "url": "file:///D:/datahub/video/frame_0003.jpg", "timestamp_ms": 66 }
      ],
      "fps": 30
    }
  ]
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `template` | object | 是 | 模板 JSON，与现有 Template.json 结构一致，从 datahub 数据库读取后直接嵌入 |
| `callback_url` | string | 是 | MVP 处理完**每个 task** 后主动 POST 结果的地址 |
| `tasks` | array | 是 | 任务列表，每个 task 独立回调 |
| `tasks[].task_id` | string | 是 | datahub 侧唯一标识，回调时原样带回 |
| `tasks[].media_type` | string | 是 | `image` 或 `video_clip`，预留扩展 |
| `tasks[].fps` | number | 否 | 视频帧率，media_type=video_clip 时建议提供 |
| `tasks[].frames` | array | 是 | 帧列表 |
| `frames[].frame_id` | string | 是 | 帧级别标识 |
| `frames[].url` | string | 是 | 文件 URL，支持 `file://`（直接读）和 `http(s)://`（下载） |
| `frames[].timestamp_ms` | number | 否 | 帧在视频中的时间位置（毫秒），单图填 0 |

#### 响应

```
HTTP 202 Accepted
```

```json
{
  "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "accepted",
  "task_count": 2
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `run_id` | string | MVP 侧生成的运行 ID，可用于追踪 |
| `status` | string | 固定为 `"accepted"` |
| `task_count` | int | 接收到的 task 数量 |

> **注意**：此接口返回不代表处理完成。处理结果通过 callback_url 异步返回。

---

### 2.2 回调通知

处理完**每一个 task** 后，MVP 主动向 `callback_url` 发起 POST 请求。

#### 成功回调

```
POST {callback_url}
Content-Type: application/json
```

```json
{
  "run_id": "mvp_a1b2c3d4",
  "task_id": "batch_20260523_001",
  "status": "completed",
  "frames": [
    {
      "frame_id": "img_001",
      "status": "completed",
      "annotation_result": {
        "image": "img_001.jpg",
        "objects": [
          {
            "bbox": [120.5, 80.3, 450.1, 520.7],
            "category": "包裹",
            "attributes": {
              "颜色": "红色",
              "有快递单": true
            },
            "confidence": 0.9234,
            "status": "accepted"
          },
          {
            "bbox": [600.2, 300.1, 780.5, 480.9],
            "category": "包裹",
            "attributes": {},
            "confidence": 0.4512,
            "status": "rejected"
          }
        ]
      },
      "runtime_trace": {
        "steps": [
          "negative:gemini-2.0-flash",
          "detect:yolo-world-v2-x",
          "verify:gemini-2.0-flash",
          "quality:opencv-heuristics",
          "attribute:gemini-2.0-flash",
          "merge:gemini-2.0-flash"
        ],
        "meta": {
          "elapsed_ms": 1250.5,
          "scene_pure_negative": false,
          "object_name": "包裹"
        },
        "candidate_history": [
          {
            "object_id": "obj_0",
            "exists": true,
            "confidence": 0.92,
            "bbox": { "x1": 120.5, "y1": 80.3, "x2": 450.1, "y2": 520.7 },
            "attributes": { "颜色": { "value": "红色", "confidence": 0.95 } },
            "quality": { "blur": { "value": "清晰", "confidence": 0.98 } }
          }
        ]
      }
    }
  ]
}
```

#### 失败回调

```json
{
  "run_id": "mvp_a1b2c3d4",
  "task_id": "clip_20260523_002",
  "status": "failed",
  "error": "FileNotFoundError",
  "detail": "Frame f001: file not found at the given URL"
}
```

#### 回调字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `run_id` | string | MVP 运行 ID |
| `task_id` | string | 对应请求时的 task_id |
| `status` | string | `completed` 或 `failed` |
| `error` | string | 失败时的错误类型 |
| `detail` | string | 失败时的详细信息 |
| `frames` | array | 每帧的推理结果 |
| `frames[].frame_id` | string | 对应请求时的 frame_id |
| `frames[].status` | string | `completed` 或 `failed` |
| `frames[].annotation_result` | object | 标注结果（成功时） |
| `frames[].runtime_trace` | object | 推理链路追踪（调试用） |

##### annotation_result.objects[].status 取值

| 值 | 含义 |
|---|---|
| `accepted` | 确认是目标物体，保留标注 |
| `rejected` | 判为负样本（非目标物体、质量不合格、或负属性命中） |
| `pending` | 不确定，需要人工复审 |

---

### 2.3 回调重试策略

回调可能因网络抖动或 datahub 服务暂时不可用而失败。MVP 采取以下重试策略：

| 重试次数 | 等待时间 | 说明 |
|---|---|---|
| 第 1 次 | 5 秒 | 立即重试，应对瞬态网络问题 |
| 第 2 次 | 30 秒 | 等待稍长，应对短暂服务中断 |
| 第 3 次 | 120 秒 | 最终重试 |
| 之后 | — | 写入本地死信队列，不再自动重试 |

**重试触发条件**（任一即可）：
- HTTP 响应状态码非 2xx（4xx 或 5xx）
- 网络超时（超时 30 秒）
- 连接失败（datahub 未响应）

**三次重试均失败后**：
- 回调结果写入 `storage/dead_letter/{run_id}/{task_id}.json`
- 记录错误日志
- 继续处理后续 task，不阻塞整批任务
- 可通过运维脚本手动重放死信队列

**幂等性说明**：
- 回调请求以 `task_id` 为幂等键
- datahub 侧建议：收到重复的 `task_id` 回调时，直接覆盖或忽略（由 datahub 业务逻辑决定）

---

### 2.4 健康检查

用于 datahub / k8s 探针检测 MVP Runtime 是否存活。

```
GET /health
```

#### 响应

```
HTTP 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok",
  "service": "mvp-runtime"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 固定为 `"ok"` |
| `service` | string | 服务标识 `"mvp-runtime"` |

---

## 3. 支持的文件 URL 协议

### file://

直接读取本地文件，零拷贝，适用于 datahub 和 MVP 在同一台机器的场景。

```
file:///D:/datahub/images/img_001.jpg
file:///C:/Users/datahub/project/config.json
```

### http:// / https://

MVP 自动下载到临时目录，处理完成后自动清理。适用于跨机器或网络文件服务场景。

```
http://datahub:8000/files/img_001.jpg
https://storage.internal/project/video_001.mp4
```

---

## 4. 处理流程

```
datahub                              MVP Runtime
  │                                       │
  │ ① POST /run_annotation_async ──────→ │
  │   {template, callback_url, tasks}     │
  │                                       │
  │ ←─ 202 Accepted {run_id}             │
  │                                       │
  │                                       │ ② 解析 template 注册属性处理器
  │                                       │ ③ 遍历 tasks:
  │                                       │    ├─ 遍历 frames:
  │                                       │    │   ├─ 解析 URL → 本地路径
  │                                       │    │   ├─ 编译计划 (Planner.compile)
  │                                       │    │   ├─ 执行推理:
  │                                       │    │   │   1. scene negative 预检（可选）
  │                                       │    │   │   2. YOLO 检测
  │                                       │    │   │   3. Gemini 验证
  │                                       │    │   │   4. OpenCV 质量分析
  │                                       │    │   │   5. Gemini 语义属性
  │                                       │    │   │   6. Gemini 负样本检查
  │                                       │    │   │   7. Gemini 合并审核
  │                                       │    │   └─ 构建 AnnotationResult
  │                                       │    └─ 构建回调帧数据
  │                                       │    │
  │                                       │ ④ POST {callback_url} ──→ │
  │   ←─ {run_id, task_id, status,       │   每完成一个 task 回调一次   │
  │        frames}                        │                             │
  │                                       │                             │
  │                                       │ ⑤ datahub 收到后：
  │                                       │    - 更新数据库状态
  │                                       │    - 存储标注结果
  │                                       │    - 通知前端/下游
```

---

## 5. URL 方案对比

| 方案 | 说明 | 适用场景 |
|---|---|---|
| `file://` | 直接读本地路径，零拷贝，零延迟 | 同机器部署（当前推荐） |
| `http(s)://` | MVP 下载到临时目录，用完清理 | 跨机器部署、对象存储 |
| 不支持 `smb://` / `nfs://` | 过于复杂，当前不考虑 | — |

---

## 6. 数据流总结

```
datahub 视角：
  数据库读取模板 → 组装成 JSON → POST 给 MVP → 等待回调

MVP 视角：
  接收 JSON → file:// 直接读图片 → 跑推理流水线 → POST 结果回 datahub

两边的耦合：
  - 仅通过 HTTP 通信
  - 不共享文件系统（file:// 只是读取不写入）
  - datahub 不需要暴露任何 API（只需接收回调）
  - MVP 不需要访问 datahub 数据库
```

---

## 7. 待实现清单

- [x] MVP 新增 `GET /health` 健康检查端点
- [ ] MVP 新增 `POST /run_annotation_async` 端点
- [ ] MVP 实现 URL 解析器（file:// + http://）
- [ ] MVP 实现异步后台任务 + 回调通知
- [ ] MVP 实现回调重试逻辑（3 次指数退避 + 死信队列）
- [ ] datahub 侧改造：拼接请求体 + 接收回调
