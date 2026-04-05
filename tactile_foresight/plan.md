一、一句话定义
预测 action chunk 执行过程中的未来触觉表征，与当前触觉联合引导扩散策略，弥补开环执行的触觉反馈盲区。支持有/无触觉传感器两种部署场景。
二、整体架构
核心设计：支持两种部署模式
模式 A：有触觉传感器
当前视觉 + 当前触觉 → 预测未来触觉 → 联合引导动作
效果最强，当前触觉辅助更准确的未来预测
模式 B：无触觉传感器
当前视觉 → 预测未来触觉 → 引导动作
优雅降级，纯视觉也能预测未来触觉
架构图
输入: 视觉 V_t, 触觉 T_t(可选), qpos

              V_t                    T_t (可选)
               │                      │
               ▼                      ▼
          DINOv2 (frozen)        DINOv2 (frozen)
               │                      │
               ▼                      ▼
          Vision Proj             Tactile Proj
               │                      │
           V_proj (256)           T_proj (256)  ← 有传感器时有值，无时为零
               │                      │
               ├──────────┬───────────┤
               │          │           │
               │    ┌─────▼─────┐     │
               │    │   TFM     │     │
               │    │ [V;T可选] │     │
               │    │  → T̂_{t+h}│     │
               │    └─────┬─────┘     │
               │          │           │
               ▼          ▼           ▼
            ┌──────────────────────────────┐
            │         TFS 评分              │
            │  输入: V_proj                 │
            │        T_proj (当前,可选)     │
            │        T̂_proj (未来,预测)     │
            │        Action                │
            │  输出: score                  │
            └──────────────┬───────────────┘
                           │
                    ∇_A score (梯度)
                           │
                           ▼
                   Base DP 去噪 + 引导
                           │
                        最终动作

核心动机：开环 chunk 的触觉盲区
扩散策略输出一个 action chunk（如 20 步动作序列），然后开环执行。推理时只能观测到当前触觉，但 chunk 执行过程中的触觉变化完全未知。如果 chunk 中途手指撞到物体、物体滑落，等到重新观测时已经来不及了。
时间线:
  t=0          t=5          t=10         t=15         t=20
  ├────────────┼────────────┼────────────┼────────────┤
  观测一次      ← 整个 chunk 开环执行，中间不观测（其实也不是,有的只取前半部分action 具体看部署情况） →
  推理一次
  输出20步动作

  t=0 只有当前触觉 → chunk 中间发生什么不知道
  TFM 预测 t+4, t+8, t+12 的触觉 → 弥补盲区

  

三、四阶段训练
Stage 0：DINOv2 视触觉对齐（1周）
目的：让视觉和触觉在同一个特征空间里
视觉图像 → DINOv2(frozen) → 768维 → Vision Proj(trainable) → 256维 ─┐
                                                                      ├→ CLIP对比Loss
触觉图像 → DINOv2(frozen) → 768维 → Tactile Proj(trainable) → 256维 ─┘

- 复用现有 clip_pretraining_xiaomi.py 的训练逻辑
- 把 modified_resnet18 替换为 DINOv2
- Projection Head: Linear(768→512) → ReLU → Linear(512→256) → L2Norm
- 只训练两个 Projection Head（~400K参数），DINOv2 frozen

1. 创建项目目录结构 — tactile_foresight/ 及子目录                                                                                                                              
2. 实现 DINOv2 特征提取器 — feature_extractor.py，封装frozen ViT-B/14                                                                                                          
3. 实现数据集 — foresight_dataset.py，构造 (V_t, T_t, T_{t+h}, qpos_t, action_t) 采样逻辑                                                                                      
验证标准：同一时刻的视觉/触觉 embedding cosine similarity > 0.6
Stage 1：TFM 触觉预见模型(视觉表征预测)（2-3周）
目的：从当前视觉（+可选的当前触觉）预测 chunk 内未来 h 步的触觉 embedding
有传感器时:
  [V_proj_t, T_proj_t] → Prediction Head → T̂_{t+h}（更准）

无传感器时:
  [V_proj_t, zero_token] → Prediction Head → T̂_{t+h}（也能用）

Prediction Head:
  horizon_tokens (3个可学习 token, 对应 h=4,8,12)
     │
     ▼
  Transformer Decoder (4层, 8头, dim=256)
    query = horizon_tokens
    key/value = [V_proj; T_proj(可选)]
     │
     ▼
  T̂_proj_{t+h} (B, 256)

Loss = MSE(T̂_proj_{t+h}, T_proj_{t+h})

关键设计：当前触觉作为可选输入——训练时随机 50% 概率 mask 掉当前触觉（用 zero token 替代），这样模型同时学会两种模式：有触觉输入时预测更准，无触觉时也能工作。h 的范围在 chunk 内（h=4,8,12 ≤ chunk_size=20），直接对应开环执行中的触觉盲区。
验证标准：预测 embedding 和真实的 cosine similarity > 0.7
Stage 2：TFS 触觉可行性评分（2周）
目的：评估"在当前触觉 + 预测未来触觉下，这个动作 chunk 合不合理"
观测侧:                                    动作侧:
[V_proj, T_proj(可选),                      Action chunk (B, 20, 7)
 T̂_proj] (B, 512~768)                          │
     │                                         ▼
     ▼                                    Conv1D(7→64→128→256) + ReLU
Linear → ReLU → Linear                   AdaptiveAvgPool1D
     │                                    Linear(256→256)
     ▼                                         │
Transformer Encoder                            ▼
(2层, 4头, dim=256)                        L2 Normalize → A_emb (B, 256)
     │                                         │
     ▼                                         │
L2 Normalize → O_emb (B, 256)                  │
     │                                         │
     └──────────── score = O · A / τ ──────────┘

训练方式：InfoNCE 双向对比学习
- 正样本: batch 对角线（同一时刻的 obs+tac 配 action）
- 负样本: batch 非对角线
- Noisy action training: 50% 概率给 action 加 DDPM 噪声
关键：TFS 接收三种触觉信号——当前真实触觉（可选）+ 预测未来触觉。当前触觉告诉你"现在接触状态如何"，预测触觉告诉你"执行这个chunk后接触会怎样"，两者互补。训练时用真实 T_{t+h}，推理时用 TFM 预测的 T̂。Noisy action training 是必须的（TouchGuide 消融显示不加从 62.5% 降到 39.2%）
验证标准：正样本 score 均值 > 0.5，负样本 < 0.15
Stage 3：基础 Diffusion Policy（1周适配）
把现有 diffusion/ 目录的 DP 做以下适配：
- encoder 从 ResNet18 换成 DINOv2（和 TFM/TFS 共享）
- action_dim = 7（joint angles）
- pred_horizon = 20（chunk_size）
- DDPM 100 步去噪
- 训练好后 frozen，不再更新
四、推理流程
# 输入: 当前视觉图像 images, 当前 qpos

# Step 1: DINOv2 提取特征（一次前向，所有模块共享）
dino_feat = dinov2(images)                  # (B, 768)
v_proj = vision_proj(dino_feat)             # (B, 256)

# Step 2: TFM 预测未来触觉
t_hat = tfm(v_proj)                         # (B, 256)

# Step 3: Base DP 编码观测条件
obs_cond = dp_obs_encoder(dino_feat, qpos)

# Step 4: DDPM 去噪 + 梯度引导
action = torch.randn(B, 20, 7)             # 纯噪声

for k in range(100, 0, -1):
    noise_pred = dp_unet(action, k, obs_cond)

    if k <= 50:  # 后50步加入引导
        action.requires_grad_(True)
        score = tfs(v_proj, t_hat, action)
        grad = torch.autograd.grad(score.sum(), action)[0]
        grad = torch.clamp(grad, -1.0, 1.0)
        noise_pred = noise_pred - eta * sqrt(1 - alpha_bar[k]) * grad
        action = action.detach()

    action = ddpm_step(action, noise_pred, k)

return action  # (B, 20, 7) 最终动作

推理计算量
组件
调用次数
单次耗时
总耗时
DINOv2 前向
1次
~30ms
30ms
TFM 前向
1次
~2ms
2ms
DP UNet 前向
100次
~3ms
300ms
TFS 前向+梯度
50次
~5ms
250ms
总计


~580ms
对于 chunk_size=20 的策略，每 20 步才推理一次，580ms 完全可接受。
五、实验设计
主实验对比
方法
类型
当前触觉
未来触觉
DP (vision only)
baseline
无
无
DP + tactile concat
feature-level fusion
真实当前
无
ACT + CLIP tactile
本项目 TactileACT
真实当前
无
RDP
stage-wise fusion
真实当前
无
DP + TouchGuide-style
当前触觉 gradient guidance
真实当前
无
DP + TactileForesight (vision-only)
未来触觉引导，无传感器
无
预测未来
DP + TactileForesight (full)
当前+未来联合引导
真实当前
预测未来
ACT + TactileForesight (full)
即插即用展示
真实当前
预测未来
四个配置展示递进关系：无触觉 → 只有当前触觉 → 只有预测未来触觉 → 当前+未来联合（最强）。每个对比都验证了一个组件的贡献。
消融实验
编号
消融内容
验证什么
A1
去掉 TFM，用 random embedding
预测未来触觉是否有用
A2
只用当前触觉 vs 当前+未来联合
未来预测的增量价值
A3
只用预测未来 vs 当前+未来联合
当前触觉对预测的辅助价值
A4
DINOv2 vs CLIP ResNet18
backbone 的影响
A5
像素 MSE vs embedding MSE
embedding 空间预测的优势
A6
不同 horizon h=4,8,12,16
最佳预测步长
A7
不同 guidance_scale η
引导强度敏感性
A8
有/无 noisy action training
噪声训练的必要性
A9
guidance vs reranking
引导方式对比
A10
TFM 输入：视觉+触觉 vs 纯视觉
当前触觉帮助预测未来触觉的程度
A11
有传感器时：预测未来触觉 vs 真实未来触觉
TFM 上限分析
六、论文"故事"逻辑
建议标题
TactileForesight: Steering Visuomotor Policies via Predicted Future Touch
故事线
第一段：问题（Why）
精细操作中，扩散策略输出一个 action chunk（如 20 步动作序列），然后开环执行。这导致三个问题：
问题1 - 触觉盲区：推理时只能观测到当前触觉，但 chunk 执行过程中（20步）的触觉变化完全未知。如果中途手指撞到物体、物体滑落，等到重新观测时已经来不及了。
问题2 - 接触前无信号：当手指接近但未碰到物体时，触觉传感器信号为零。而很多任务中"第一下接触"最关键（开锁、穿鞋带），接触前恰恰是最需要引导的时刻。
问题3 - TouchGuide 也只用当前触觉：即使是最先进的 TouchGuide，也只能基于当前触觉引导（reactive），无法预判 chunk 内未来的接触事件。
核心问题：能不能在输出 action chunk 之前，就预测 chunk 执行过程中会产生的触觉，从而提前规避不良接触？
第二段：观察（What we found）
两个关键 insight：
1. 视觉包含触觉的先兆信息。人在伸手去抓一个玻璃杯之前，通过视觉就能预判：杯子是光滑的、需要轻拿。视觉中物体的形状、材质、手的接近速度，都暗示着未来的触觉。当前触觉更能辅助预测——正在触碰的感受帮助预判下一步会怎样。
1. 在预训练特征空间中预测表征比预测像素有效得多（FRAPPE, DINO-WM 已验证）。不需要预测触觉图像的每个像素，只需要预测一个语义 embedding。
第三段：方法（How）
TactileForesight 框架三个模块：
- TFM（Tactile Foresight Model）：从当前视觉 + 当前触觉（可选）预测 chunk 内 h 步后的触觉 embedding。有传感器时用当前触觉辅助预测更准确，无传感器时纯视觉也能工作。
- TFS（Tactile Feasibility Score）：对比学习评分器，输入当前触觉（可选）+ 预测未来触觉 + 候选动作，输出可行性分数。当前触觉判断"现在状态如何"，预测触觉判断"执行后会怎样"。
- 推理时梯度引导：基础 DP 去噪过程中注入 TFS 梯度信号，不修改基础策略训练，即插即用。
关键特性：支持有/无触觉传感器两种部署模式。有传感器时当前+未来联合引导（效果最强），无传感器时纯视觉预测未来触觉引导（优雅降级）。
第四段：实验（Results）
核心结果（四个配置递进展示）：
- DP baseline → DP + TactileForesight(full)：大幅提升，证明整体框架有效
- DP + TouchGuide-style → DP + TactileForesight(full)：加了未来预测后超过只用当前触觉，证明 foresight 的价值
- TactileForesight(vision-only) vs TactileForesight(full)：有传感器时更强，但纯视觉也能用
- 首次接触任务上优势最明显：在接触前触觉为零的场景中，foresight 的优势最大
第五段：贡献（Contributions）
1. 开环 chunk 的触觉预见：弥补 chunk 执行中间没有触觉反馈的盲区
2. 跨模态跨时间预测：[视觉+当前触觉] → 未来触觉，文献中前所未有
3. Proactive 而非 Reactive：接触前就预判，不是碰到了才反应
4. 灵活部署：支持有/无传感器两种模式，即插即用搭配任意策略
和 TouchGuide 的定位关系
不对立，站在它肩膀上："TouchGuide 优雅地证明了推理时触觉引导的有效性。我们沿袭其 inference-time steering 范式，将触觉信号从当前感知扩展到未来预见——在保留当前触觉引导能力的同时，增加了对 chunk 内未来接触的预判。TactileForesight 是 TouchGuide 在时间维度上的自然延伸。"
七、时间线
周
内容
产出
1
DINOv2 环境搭建 + Stage 0 对齐训练
对齐好的 Projection Heads
2-3
Stage 1 TFM 实现与训练
TFM checkpoint
4-5
Stage 2 TFS 实现与训练
TFS checkpoint
6
Stage 3 DP 适配 DINOv2 + 训练
Base DP checkpoint
7-8
推理时引导实现 + 调参
完整推理 pipeline
9-12
实验（主实验 + 消融 + 可视化）
全部实验数据
13-16
论文撰写
投稿
八、相关论文
优先级
论文
链接
★★★
TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance
https://arxiv.org/abs/2601.20239
★★★
FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment
https://arxiv.org/abs/2602.17259

★★★
DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance
https://arxiv.org/abs/2506.13922
★★★
On the Guidance of Flow Matching
https://arxiv.org/abs/2502.02150
★★☆
DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
https://arxiv.org/abs/2411.04983
★★☆
Visuo-Tactile World Models
https://arxiv.org/abs/2602.06001
★★☆
Latent Policy Barrier
https://arxiv.org/abs/2508.05941
★★☆
Ensuring Force Safety in Vision-Guided Robotic Manipulation via Implicit Tactile Calibration
https://arxiv.org/abs/2412.10349
★☆☆
Sparsh: Self-supervised Touch Representations for Vision-Based Tactile Sensing
https://arxiv.org/abs/2410.24090