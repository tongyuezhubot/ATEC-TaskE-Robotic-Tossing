# 服务器时序世界模型交接（2026-10-08）

实际工作目录：/root/Desktop/ATEC_TaskE_Standalone。本次修改直接在服务器完成。
既有 ACT 未提交修改、成功数据 ExternalLinks、录像与 checkpoint 保留。
修改前源文件备份：work/remote_before_recurrent_3xoZzAo0/source_before.tar.gz。
既有 Git diff 与 ACT 文件哈希保存在同目录；两份 ACT 文件哈希已再次确认相同。

## 数据与动作事实

combined_20 包含20条轨迹，68467步，ExternalLinks仍能正常解析。
示范动作实际范围约 [-2.9418, 2.9404]，继续使用服务器原有 ±3.0 限制。
关节相对位置减默认位置；关节相对速度减默认速度；第三段为上一环境动作。
Task E 关闭观测噪声。旧记录第一帧之前的warm-up动作未知，训练跳过首个转移。
旧数据没有逐步奖励、终止和事件，不再伪造最后可见帧的成功奖励。
旧成功属性只用于轨迹级数据选择，不能当成逐步终止标签。

## 关键文件

- demo/world_model_recurrent.py：复用服务器现有编码器、动力学、actor和辅助头，增加GRU、EMA目标、下一本体/视觉特征预测、低分辨率视觉锚定、方差约束和7维事件头。
- demo/world_model_sequence.py：连续窗口，按整条轨迹留出；前4帧重建belief，后续帧监督；旧数据未知reward/events有明确mask。
- demo/world_model_sequence_replay.py：单环境连续序列Replay，环形覆盖和episode边界校验；在线失败动作不作为专家actor监督。
- scripts/world_model/aligned_transitions.py：在Isaac自动reset前捕获下一24维/RGB观测与reward事件计数；训练真值隔离于training_labels，reset hook每次step后恢复。
- scripts/act/task_e/collector.py：--aligned增加明确转移，保留已有global IK专家；失败与超时可保留。专家阶段仅作为标签。
- scripts/act/collect_demos_task_e.py：拒绝覆盖已有数据；新格式兼容既有ACT obs/actions/images入口。
- demo/world_model_dataset.py：移除旧数据伪造奖励和伪造末端done。
- demo/world_model_prior_planner.py、demo/solution_world_model.py：时序部署/reset、actor先验附近短视野CEM和动作变化限制。旧checkpoint可继续加载；新checkpoint没有真实outcome监督时，CEM会明确拒绝，actor模式可诊断先验。
- scripts/world_model/pretrain_recurrent.py：小批序列，固定留出窗口选checkpoint；旧权重仅迁移编码/动作先验，不迁移曾使用伪造label的reward/value头。
- scripts/world_model/train_recurrent.py：最多5000步，默认1000步，每10步更新一次，50%专家序列混合；小幅平滑扰动，保存带事件轨迹与独立checkpoint。
- scripts/play_atec_task.py：录像无关的max_steps、seed与策略reset。
- scripts/world_model/run_recurrent.sh：可复用运行命令。

## 执行顺序

从服务器项目目录运行：

    bash scripts/world_model/run_recurrent.sh pretrain
    bash scripts/world_model/run_recurrent.sh collect-small
    bash scripts/world_model/run_recurrent.sh train-labeled

先用旧20条数据学习时序表示/动作先验，再用3条新标签数据补监督。
小批采集不能保证包含抓空、滑落、投偏和恢复；先看事件记录，缺哪类再少量补采。
当前未新增自动恢复专家/恢复策略，也没有按旧帧伪造失败分类。

确认动作先验能驱动合理动作后才执行：

    bash scripts/world_model/run_recurrent.sh online
    bash scripts/world_model/run_recurrent.sh eval

默认在线1000步，控制步0.02秒对应20秒仿真时间；墙钟成本按实际吞吐估计。
CEM默认6步视野、64候选、2轮，只执行第一个动作后更新观测。
有新修改或新失败证据后才做下一次固定预算闭环。

## 能力和证据边界

12项必要单元测试已在服务器已有conda环境通过；语法通过。
真实20条数据已完成一次GPU小批次更新，按16条训练/4条留出划分。
train total=0.83259，held-out total=0.74025（仅一次更新，不能据此评价任务效果）。
产物：work/recurrent_smoke_xYAPc4gq/smoke.pt；日志在同目录train.log。
真实数据小批训练产物单独保存在work/recurrent_smoke_*，只用于工程往返验证。
历史ACT单次9分仍是已知结果，未触发basket_success。
未声称已提高抓取/投放/恢复成功率，没有新的完整闭环成绩。
reset前采集hook已用模拟auto-reset测试；真实Isaac路径须在首批collect-small确认。
事件区分质量、价值可靠性、实时推理耗时与跨seed成功率仍未知。

## 第二批推进：真实采集标签修复

用户已完成pretrain 1000步：最佳模型在step800，validation total=0.1193216；非闭环成绩。
真实aligned_003共3轨迹10550步：traj_0/traj_2触发basket_success；traj_1未成功。
发现Isaac 5.0 TerminationManager的get_term/get_active_iterable_terms返回历史终止缓存。
因此原始traj_1的3700步、traj_2的前3426步success标签错误，原文件完整保留。
未来采集改为直接调用已配置的原任务成功谓词，不读历史done缓存。
repair_aligned_labels.py依据保存的本地物体位置和原任务相同float32几何边界重建success。
修正视图：datasets/atec_task_e/aligned_003_corrected/trajectory.hdf5。
图像/动作等ExternalLinks仍依赖aligned_003原文件，禁止移动或删除原文件。
traj_1作为未成功/采集截断保留，但actor_supervision=False；不伪造任务终止失败。
sequence loader拒绝无可靠标签来源标记的旧aligned文件，避免误用原始串标签数据。
预训练新增早停：默认每100步验证，patience3、min_delta0.001、min_steps300；
train-labeled每50步验证，最多300步、min_steps100。真实最佳损失仍独立用于保存checkpoint。
带事件数据50%均匀窗口+50%真实正事件窗口；混合源各占50%；验证固定加入正事件窗口。
事件稀少，3轨迹不能证明泛化、失败恢复或可靠的价值估计。CEM闭环能力仍需真实验证。

修正和训练命令（只首次执行，已有文件拒绝覆盖）：

    /opt/conda/envs/isaaclab/bin/python scripts/world_model/repair_aligned_labels.py --source datasets/atec_task_e/aligned_003/trajectory.hdf5 --output datasets/atec_task_e/aligned_003_corrected/trajectory.hdf5
    bash scripts/world_model/run_recurrent.sh train-labeled

## 本轮真实运行结果与停止点（已执行，不要重复运行）

1. 数据修正已完成。修正7126个success标签；原aligned_003没有写入。
   数据几何success与terminated逐步完全一致：traj_0末步、traj_2末步为True，
   traj_1全False。原TaskE的实际TABLE_TOP_Z=0.8266426479059952。
2. 修复前代码备份：work/before_label_fix_evdRujBC/source_before.tar.gz。
   14项原/新增测试通过；后续冻结先验的定向更新回归测试也通过。
   测试证明确实有结果头梯度，更新后全部部署先验参数和actor输出不变。
3. 第一次联合带标签训练已完成300步，留出total从0.635323降到0.161865。
   日志：work/labeled_train_T21V4Xv4/train.log。
   checkpoint：atec_robot_model/baseline/world_model/recurrent_labeled_300.pt。
4. 对这个联合训练版本只跑了一次4000步、seed43的CEM真实闭环：
   score=0，basket_success=False，max_steps结束；并非有效任务策略。
   结果：runs/recurrent_labeled_eval_0v3XFfyJ/result.json，
   日志：同目录play.log。没有重复仿真，没有运行online增预算。
5. 只用已有traj_2留出数据的9段固定窗口（252个有效帧）诊断：
   原先验action RMSE=0.061671，联合训练后=0.144284；
   重复上一动作基线=0.013549。说明总损失下降掩盖了动作先验退化，
   但这不是对整个闭环失败原因的完整证明。
   脚本：scripts/world_model/diagnose_recurrent_prior.py；
   结果：runs/recurrent_labeled_eval_0v3XFfyJ/prior_diagnosis.json。
6. 已针对该退化修复：--freeze_prior仅训练reward/value/event头，
   不更新部署encoder、GRU、dynamics、actor；忽略无标签legacy源采样，
   同时保留原训练/留出划分。在线恢复此类checkpoint也会保持先验冻结。
7. 冻结先验版本已完成300步：留出total=0.270509，events=0.143323；
   actor验证项始终0.001291406，未被结果监督拖坏。还未闭环测试。
   新候选：atec_robot_model/baseline/world_model/recurrent_labeled_heads_300.pt，
   日志：work/labeled_heads_train_yuBCYvVn/train.log。
   run_recurrent.sh的train-labeled与online已切换此新候选；旧失败checkpoint保留。
   两轮都持续显著改善，因此没有触发早停；不是早停失效。
8. play_atec_task.py新增--result_file、每500步进展与显式basket_success结果，
   只用于评测诊断，不给策略输入任何真值、得分、奖励或成功标记。
   shell入口固定使用既有isaaclab Python环境，无环境安装变更。

下一步优先做原动作先验的单次有上限诊断，而不是直接online训练：

    bash scripts/world_model/run_recurrent.sh prior-eval

这是1500步actor模式检查，只能看先验能否推进抓取，不能证明三物体完整成功。
原actor与新冻结候选部署先验相同，结果预测头不参与actor模式动作。
若先验不能推进，先处理动作模仿/闭环分布问题；不凭奖励预测损失低就加大CEM/训练预算。
冻结候选的CEM可靠性、跨seed能力、失败恢复仍未知；最终目标尚未完成。

## 控制交接诊断与修复（本轮已完成，勿重复长评测）

用户提供actor-only 1500步结果：seed43、score0、未成功。
本轮只运行修复前/修复后各300步，共600步真实仿真；没有新训练或长评测。
增加--control_trace保存每步合法24维观测、动作、内部控制阶段、home误差和belief范数。
这些日志仅用于诊断，专家阶段/物体真值/奖励/成功仍不进入部署策略。

修复前：第32步actor已接管，排除home卡住；接管最大关节速度0.841665rad/s，
之后动作漂移，最大相邻动作变化约0.527577。不是站着不动。
还存在训练前4帧仅burn-in、部署却直接使用冷belief，以及上一动作初始化不一致。
20条旧示范初始命令第5维均约0.44，旧home再编码为0.0398。
已将固定初始化命令与已有示范对齐，不用部署物体真值。

改动：
- demo/solution_world_model.py：home位置/速度稳定后接管；4帧真实观测预热；
  arm动作每步变化<=0.05，finger<=0.02；home300步不收敛会明确giveup；
  reset清除home/burn-in计数。DEMO_HOME_ACTION与home_is_stable供在线入口复用。
- scripts/play_atec_task.py：--control_trace独立JSON和阶段切换日志，不覆盖已有文件。
- scripts/world_model/train_recurrent.py：匹配home稳定/记忆预热规则，
  actor/CEM接管前不加探索噪声，动作变化同样受限。只做语法检查，本轮未运行online。
- tests/test_world_model_handoff.py：3项定向测试通过，验证稳定阈值、
  首次actor前4帧真实记忆、动作限制、reset和home超时。
- scripts/world_model/run_recurrent.sh：新增control-check入口，固定300步、有独立日志目录。

修复后实测：47-50步预热，51步actor接管；
接管home最大误差0.0014165rad、最大速度0.0149455rad/s；
真实arm最大动作变化0.05000007、finger0.019999996（浮点误差范围）。
仍score0，随后偏离示范；短测试只证明交接和限幅成立，不证明抓取/恢复有效。
原始trace：runs/recurrent_control_trace_irA2sNPC/control_trace.json。
修复trace：runs/recurrent_handoff_fixed_36zFFyT4/control_trace.json。
量化汇总：同目录handoff_analysis.json。
本轮源备份：work/before_control_trace_OfXWOJk0/source_before.tar.gz，
其solution_with_trace.py保留了仅加日志、尚未改变交接的版本。

只用traj_2前300步真实示范做记忆长度对照：
continuous actor RMSE=0.0745748；rolling12=0.0749080；
重复上一动作基线=0.0192784。两种记忆差异不足以支持其是主要原因。
示范初始100步中actor仍有约0.0797误差，说明当前动作先验不够准确；
不能据此保证“再训练更多步”会解决闭环问题。
环境video_rgb是normalize=False的RGB，未发现float归一化误配证据。

当前停止点：初始化一致性问题已修；当前动作先验仍不能视为有效任务策略。
不重跑原pretrain，不直接online扩大预算；下一步应针对动作先验的监督/拟合与
闭环分布偏差设计小预算改动，再做一次对应验证，而不是继续只训练结果头。
保留所有既有ACT、数据、录像和checkpoint。本轮没有产生新checkpoint。

后续确有控制/模型改动才复现短检查：

    bash scripts/world_model/run_recurrent.sh control-check

此命令不是完整任务评测；现有两次短检查已落盘，当前不需要再运行。
