# 恢复数据批次交接：2026-10-08

服务器项目：/root/Desktop/ATEC_TaskE_Standalone。
本轮任务是增加小批真实失败/纠正过程，不是再次扩大模型训练预算。
原ACT两份既有未提交改动、原数据ExternalLinks、录像和全部旧checkpoint保留。
代码改动前备份：work/before_recovery_collect_yausAfpN/source_before.tar.gz；
其中existing.diff记录了原有未提交内容，pretrain_before.py/model_before.py保留后续训练入口改动前版本。

## 实际采集结果

新数据：datasets/atec_task_e/aligned_recovery_006/trajectory.hdf5。
审计：同目录audit.json。日志：work/recovery_collect_jY25LvVg/collection.log。
seed104，三物体保持3→2→1顺序；正常reset后的位置随机化遵循已有spawn范围。
六条均保留，26,550步、96×96 RGB；2条全任务basket_success、4条未成功。

| 轨迹 | 场景 / 目标物体 | 步数 | 目标重抓进篮 | 全任务成功 |
| --- | --- | ---: | --- | --- |
| traj_0 | 错位抓取 / 香蕉3 | 4223 | 是 | 是 |
| traj_1 | 错位抓取 / 芥末2 | 4000 | 否 | 否 |
| traj_2 | 低高度松爪 / 香蕉3 | 4800 | 是 | 否：糖盒后来未成功 |
| traj_3 | 低高度松爪 / 芥末2 | 4000 | 否 | 否 |
| traj_4 | 投偏 / 香蕉3 | 4727 | 是 | 是 |
| traj_5 | 投偏 / 糖盒1 | 4800 | 否 | 否：芥末也未成功 |

这些是脚本专家（训练数据生成器）的结果，不是世界模型/ACT部署成绩。
当前学习策略仍未证明basket_success或失败恢复有效。
低高度松爪是人为受控扰动，不等同于自然滑落；没有抛掷动作、没有在episode中瞬移物体。
只变动正常EE目标/夹爪命令；原任务物理、reward、termination、相机与动作接口保持。
恢复组不覆盖actuator配置；原Piper默认仍为800 stiffness/80 damping。
每物体最多重试一次；每条最多6000记录步；整个批次最多6条、6次外层尝试。
达到上限保留失败，不无期限收集、不额外增加尝试。

## 真实监督与隔离

policy_input/next_policy_input仅包含合法24维proprio和RGB。
reward、terminated、truncated、next_obs_valid来自对齐采集协议；
重抓不会reset环境/一次性reward计数；环境真正终止前捕获末帧。
收集器停止而环境未终止时，末帧标truncated=True，不伪造task terminal。
task basket_success逐帧独立调用原配置谓词，不读历史done缓存。
保存的object_pos属于训练真值，不作为部署输入。

training_labels新增active_object、recovery_attempt、fault_request、fault_applied、actor_valid。
recovery_report属性记录失败观察/重试/进篮事件，含步骤索引。
fault_request是计划扰动类型，不能直接当作已发生失败的标签。
failure_observed是专家几何诊断，不是原TaskE的官方失败终止标记。
只有真实几何确认进篮的物体尝试可用作BC；目标物体的首轮受控失误全部排除，
即使整条轨迹最终成功，也不让模型模仿这些失误。
整体失败轨迹中已完成其他物体的正确动作仍可用；未完成尝试全部masked out。

新批次actor_valid共15,650步；10,900步不作为动作模仿监督，
其中5,500步是目标物体的首轮受控失误尝试。
所有26,550步仍可用于真实动力学/reward/events监督。
旧ACT训练器没有消费actor_valid：不要把这个恢复批次直接喂给旧ACT训练命令。
world_model_sequence.py会读取掩码，缺失必需掩码时拒绝加载。
部署predicts仍只用24维/RGB，不读这些训练标签。

## 关键变更

- scripts/act/task_e/recovery.py：三类受控失误、支撑面/空间检查、每物体一次重试、回填验证后的actor mask。
- scripts/act/task_e/state_machine.py：明确retry_object/skip_object生命周期；先开爪回HOME等待，再用当前物体姿态重算抓取。
- scripts/act/task_e/collector.py：保留原global IK，追加恢复路径、记录步数上限、训练标签、采集截断。
- scripts/act/cli_args.py、collect_demos_task_e.py：--recovery_mix、--max_record_steps，要求保留失败；保存报告/掩码。
- scripts/world_model/aligned_transitions.py：新增标签仍隔离在training_labels。
- demo/world_model_sequence.py：逐步actor_valid与episode级监督许可相交。
- scripts/world_model/audit_recovery_data.py：一次性全本体时序/动作/成功几何/episode边界/掩码审计；
  图像相邻检查为每条12个固定样本，不声称逐像素审核全部帧。
- scripts/world_model/pretrain_recurrent.py：新增--actor_only，只更新actor，
  冻结encoder、GRU、dynamics、reward/value/events；动作变化帧权重5，其他帧权重1；
  按固定留出actor损失选最佳checkpoint，支持早停，不覆盖已有权重。
- demo/world_model_recurrent.py：仅训练损失接受actor_weight；未改变部署输入/网络结构。
- run_recurrent.sh：collect-recovery、audit-recovery、refine-actor入口。

10项相关恢复/序列测试通过；动作头单独更新和原序列往返2项定向测试通过。
测试仅用小型随机模型；没有对真实数据启动新训练，没有生成新checkpoint。
actor_only不会重新校准结果头，outcome_supervised沿用init权重的真实监督状态；
因此新候选应先actor模式验证，不能凭数据增加就宣称CEM可靠。

## 本轮已执行：不要重复

    bash scripts/world_model/run_recurrent.sh collect-recovery
    bash scripts/world_model/run_recurrent.sh audit-recovery

两者已有产物，再运行会拒绝覆盖。直接看audit.json即可。
原combined20、aligned_003、aligned_003_corrected全部保留。

## 下一次小预算动作先验精修（已准备，尚未运行）

    bash scripts/world_model/run_recurrent.sh refine-actor

最多400步，batch4/sequence12；每100步验证，patience3，
min_delta=1e-5、min_steps200；按actor验证损失选best，不再按总损失掩盖动作退化。
使用旧20条成功与新6条混合；seed42按完整episode划分，与旧20条原留出保持一致。
新产物（尚不存在）：atec_robot_model/baseline/world_model/recurrent_actor_recovery_400.pt。
先确认actor留出拟合/动作变化帧质量有改善，再决定是否值得一次真实闭环；
不自动启动online或CEM，不宣称400步足够。

候选的短控制检查入口（只在新候选确已训练且有改进后运行一次）：

    ATEC_TASK_E_WORLD_MODEL=/root/Desktop/ATEC_TaskE_Standalone/atec_robot_model/baseline/world_model/recurrent_actor_recovery_400.pt ATEC_WM_CONTROL=actor bash scripts/world_model/run_recurrent.sh control-check

control-check只有300步，不能证明三物体完整成功。已有旧模型0分证据不要重复长评测。
当前主要缺口：芥末/糖盒的成功恢复样本不足；朝向覆盖未做专门扩展；
学习策略的闭环有效性、自然失败恢复和跨seed泛化仍未知。
后续若补数据，应补这些明确缺口，而不是盲目扩大同类成功样本。

## 动作范围兼容补充

action_ranges.json记录了本批次实际命令范围：[-2.9346228, 3.3732624]。
15,650个actor_valid帧中433帧有至少一维超出旧模型的±3。
原始动作和actor_valid没有因此被裁剪或改写；±3是旧世界模型输出范围，不是环境动作裁剪。
refine-actor现在明确使用--actor_action_limit 3.5，只对新候选生效。
旧checkpoint/旧推理配置仍为原值，不在原地修改。
候选config与actor、inverse辅助头的ScaledTanh范围同步，保证保存/加载后的行为一致；
inverse辅助头的参数仍冻结，输出范围随candidate config变化。
加入一个定向保存/加载测试并通过；本轮仍未启动真实训练或生成候选checkpoint。
新范围覆盖本批数据，但并不证明所有未来恢复姿态都覆盖；动作每步变化保护仍有效。
