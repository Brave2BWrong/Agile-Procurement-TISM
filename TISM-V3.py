import os
import sys

# =====================================================================
# 一键运行拦截器：必须放在代码的最开头！自动唤醒 Streamlit 并跳过提示
# =====================================================================
if not os.environ.get("STREAMLIT_RUNNING"):
    os.environ["STREAMLIT_RUNNING"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    import subprocess

    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    sys.exit()

# ------------------ 真正的业务代码 ------------------
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 页面基本设置：宽屏，方便阅读长表格与滚动长页面
st.set_page_config(page_title="刘总论文 - 敏捷采购 M-TISM 推演系统", layout="wide", initial_sidebar_state="expanded")

# ================= 业务默认值数据初始化 =================
FACTORS = [
    "F1 组织文化阻力", "F2 权力失控焦虑", "F3 层级权力固化",
    "F4 敏捷人才缺口", "F5 供应链信任缺失", "F6 价值认知错位",
    "F7 静态绩效约束", "F8 事前计划固化", "F9 变更弹性匮乏",
    "F10 敏捷迭代失效", "F11 协同实操脱节", "F12 交付底线维稳",
    "F13 应急响应不足", "F14 增量价值缺失"
]
N = len(FACTORS)

# 1. 专家关系矩阵 (SSIM) 默认值
default_ssim = [['O'] * N for _ in range(N)]
default_ssim[0][1], default_ssim[0][2], default_ssim[0][5] = 'V', 'V', 'V'
default_ssim[1][2], default_ssim[1][7], default_ssim[1][8] = 'V', 'V', 'V'
default_ssim[2][6], default_ssim[2][7] = 'V', 'V'
default_ssim[3][9], default_ssim[3][10] = 'V', 'V'
default_ssim[4][8], default_ssim[4][10] = 'V', 'V'
default_ssim[5][4], default_ssim[5][6] = 'V', 'V'
default_ssim[6][7] = 'V'
default_ssim[7][8], default_ssim[7][11] = 'V', 'V'
default_ssim[8][12] = 'V'
default_ssim[9][12], default_ssim[9][13] = 'V', 'V'
default_ssim[10][9] = 'V'
default_ssim[12][13] = 'V'

# 2. 初始关系业务解释字典 (Interpretive Logic)
default_logic = {
    (0, 1): "传统控制型组织文化根深蒂固，直接导致管理层对敏捷放权后可能产生的失控抱有极大焦虑。",
    (0, 2): "保守的文化氛围促使组织倾向于维持现状，固化了现有的层级权力分配机制。",
    (0, 5): "缺乏敏捷土壤的文化导致全层级对敏捷的战略价值认知产生错位，仅将其视为一线战术工具。",
    (1, 2): "管理层出于合规避险心理，更倾向于将审批权集中，加剧了权力的固化。",
    (1, 7): "对跨境风险的担忧促使管理层要求制定极度详尽且不可变的事前采购计划，以求心理安全。",
    (1, 8): "因担心放权导致违规，严格限制事中变更权限，导致现场应对突发工况的弹性匮乏。",
    (2, 6): "自上而下的集权体制往往配套单一的、注重底线防范的传统静态绩效考核机制。",
    (2, 7): "层层审批的权力架构决定了采购动作必须严格按照预先获批的死板计划执行，难作调整。",
    (3, 9): "缺乏兼具国际工程专业与敏捷思维的复合人才，使得短周期迭代等硬核敏捷工具无法落地。",
    (3, 10): "人员能力不足导致“理解敏捷但不会操作”，产生了协同意识与业务实操的严重脱节。",
    (4, 8): "采供双方缺乏战略信任，多采用零和博弈与刚性合同，剥夺了执行过程中的变更调整弹性。",
    (4, 10): "信任赤字阻碍了与供应商的深度业务融合，使得敏捷协同流于表面沟通，无法落实到实操。",
    (5, 4): "未能认识到敏捷供应链的长期共赢战略价值，导致无法与核心供应商建立深度信任与关系契约。",
    (5, 6): "只看重底线管控价值而忽视敏捷增量价值创造，导致考核体系沿用传统静态防守型指标。",
    (6, 7): "以防范违规为主的静态绩效考核，迫使执行层在制定计划时留足余量并高度固化以求自保。",
    (7, 8): "过度详尽且固化的事前计划，在实际采购执行中一旦遇到变更，完全没有敏捷调整的余地。",
    (7, 11): "刚性计划虽僵化，但能通过严格层级卡点勉强守住成本、合规等项目交付的最低底线。",
    (8, 12): "一旦发生国际物流阻断等突发事件，缺乏变更弹性导致审批冗长，应急响应能力严重滞后。",
    (9, 12): "无法将大型长周期采购拆解为短周期迭代，面对风险时整个计划只能停滞，无法敏捷响应。",
    (9, 13): "敏捷迭代的失效意味着无法在执行中持续交付和优化，从而彻底丧失了敏捷所能带来的增量价值。",
    (10, 9): "一线团队虽然有意愿但缺乏实操落地方法与工具，导致敏捷理念无法转化为实际的短周期迭代。",
    (12, 13): "应急响应能力的缺失，使得项目团队只能疲于应付危机保底，根本无力去创造超出预期的增量价值。"
}

# ================= 页面头部 =================
st.title("🏗️ 国际工程采购敏捷治理 M-TISM 交互式推演系统")
st.markdown("""
> 本系统严格对标 ***Building theory of sustainable manufacturing using total interpretive structural modelling*** 论文规范。  
> 结合《刘总论文》中提取的核心实证数据，实现敏捷采购痛点的量化降维与可视化重构。
""")
st.markdown("---")

# ================= 一、 输入区域 (Inputs) =================
st.header("📥 第一部分：系统输入 (Inputs)")
st.markdown("包含了论文方法论中要求的**因素列表**、**SSIM 矩阵**以及**业务解释性文本**。")

col_in1, col_in2 = st.columns([1, 2.5])
with col_in1:
    st.subheader("1. 因素列表 (Variables/Enablers)")
    st.info("基于实证问卷数据识别出的 14 个采购治理核心因素。")
    df_factors = pd.DataFrame({"编号": [f"F{i + 1}" for i in range(N)], "因素名称": FACTORS})
    st.dataframe(df_factors, hide_index=True, use_container_width=True, height=450)

with col_in2:
    # ---------------- 优化点：使用 Form 表单拦截自动刷新 ----------------
    with st.form("ssim_input_form"):
        st.subheader("2. 专家判定关系矩阵 (SSIM)")
        st.info(
            "专家对14个因素两两之间关系的判断（V, A, X, O）。这是程序最核心的原始数据输入。**（双击表格可直接修改，修改后请点击下方运行按钮）**")
        df_ssim = pd.DataFrame(default_ssim, columns=[f"F{i + 1}" for i in range(N)],
                               index=[f"F{i + 1}" for i in range(N)])

        # 允许用户在表单内任意修改
        edited_df = st.data_editor(df_ssim, use_container_width=True, height=430)

        # 明确的运行按钮
        submitted = st.form_submit_button("🚀 确认输入参数并重新生成 M-TISM 全套输出", type="primary")

# 如果用户点击了运行按钮，给出视觉反馈
if submitted:
    st.toast("✅ 数据已接收，正在重新计算 M-TISM 模型...")

# ================= 核心算法演算 =================
# 1. 构建初始可达矩阵
adj_matrix = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(i + 1, N):
        val = str(edited_df.iloc[i, j]).upper()
        if val == 'V':
            adj_matrix[i][j], adj_matrix[j][i] = 1, 0
        elif val == 'A':
            adj_matrix[i][j], adj_matrix[j][i] = 0, 1
        elif val == 'X':
            adj_matrix[i][j], adj_matrix[j][i] = 1, 1
np.fill_diagonal(adj_matrix, 1)

# 2. 计算最终可达矩阵 (传递闭包 Warshall)
reach_matrix = adj_matrix.copy()
transitive_links = []
for k in range(N):
    for i in range(N):
        for j in range(N):
            if reach_matrix[i][j] == 0 and reach_matrix[i][k] == 1 and reach_matrix[k][j] == 1:
                reach_matrix[i][j] = 1
                transitive_links.append((i, j, k))

# 计算驱动力与依赖度
drive_pwr = np.sum(reach_matrix, axis=1)
dep_pwr = np.sum(reach_matrix, axis=0)

# 为前端展示准备带 "1*" 的最终可达矩阵 (学术论文标准展示法)
final_display_matrix = []
for i in range(N):
    row = []
    for j in range(N):
        if adj_matrix[i][j] == 1:
            row.append("1")
        elif reach_matrix[i][j] == 1:
            row.append("1*")  # 标识传递性
        else:
            row.append("0")
    row.append(str(drive_pwr[i]))  # 附加驱动力
    final_display_matrix.append(row)

# 3. 层级划分算法
levels_dict = {}
level_records = []
remaining = list(range(N))
current_lvl = 1
while remaining:
    reach_sets = {i: set([j for j in remaining if reach_matrix[i][j] == 1]) for i in remaining}
    ante_sets = {i: set([j for j in remaining if reach_matrix[j][i] == 1]) for i in remaining}
    inter_sets = {i: reach_sets[i].intersection(ante_sets[i]) for i in remaining}

    top_factors = [i for i in remaining if reach_sets[i] == inter_sets[i]]
    if not top_factors: top_factors = remaining.copy()

    for f in top_factors:
        level_records.append({
            "因素 (Factor)": FACTORS[f],
            "可达集 (Reachability Set)": ", ".join([f"F{j + 1}" for j in reach_sets[f]]),
            "先行集 (Antecedent Set)": ", ".join([f"F{j + 1}" for j in ante_sets[f]]),
            "交集 (Intersection Set)": ", ".join([f"F{j + 1}" for j in inter_sets[f]]),
            "所处层级 (Level)": f"Level {current_lvl}"
        })
    levels_dict[current_lvl] = top_factors
    for f in top_factors: remaining.remove(f)
    current_lvl += 1

st.markdown("---")
# ================= 二、 输出区域 (Outputs) =================
st.header("📤 第二部分：系统输出 (Outputs)")
st.markdown("严格对应学术文献标准，基于您上方的最新参数，实时生成完整的 6 大输出模型与图表。")

st.subheader("输出 1：初始可达矩阵 (Initial Reachability Matrix)")
st.info("将 SSIM 转换后的 0/1 矩阵，仅包含专家直接认定的因果关系。")
df_init = pd.DataFrame(adj_matrix, columns=[f"F{i + 1}" for i in range(N)], index=[f"F{i + 1}" for i in range(N)])
st.dataframe(df_init, use_container_width=True)

st.subheader("输出 2：最终可达矩阵 (Final Reachability Matrix)")
st.success(
    "包含通过 Warshall 算法自动推导出的“传递性链接”（标记为 **1***），同时在末列和末行计算出了驱动力 (Driving Power) 和依赖度得分 (Dependence)。")
df_final = pd.DataFrame(final_display_matrix, columns=[f"F{i + 1}" for i in range(N)] + ["驱动力 (Driving)"],
                        index=[f"F{i + 1}" for i in range(N)])
df_final.loc['依赖度 (Dependence)'] = list(dep_pwr) + ["-"]
st.dataframe(df_final, use_container_width=True)

st.subheader("输出 3：层级划分表 (Level Partitioning Table)")
st.warning("算法详细演算过程：当“可达集”与“交集”完全相同时，该因素被抽取作为当前最高层（Level）。")
st.dataframe(pd.DataFrame(level_records), use_container_width=True, height=350)

st.markdown("---")
col_vis1, col_vis2 = st.columns([1.2, 1])

with col_vis1:
    st.subheader("输出 4：有向图结构模型 (Structural Model)")
    st.info("可视化的层级图，直观展示因素间的因果链条。\n- 🖤 **实线**：直接关系\n- 🔘 **虚线**：隐性传递关系")

    dot = graphviz.Digraph(engine='dot')
    dot.attr(rankdir='BT', size="9,9")
    for lvl, f_list in levels_dict.items():
        with dot.subgraph() as s:
            s.attr(rank='same')
            for f in f_list:
                s.node(str(f), f"F{f + 1}\n{FACTORS[f].split(' ')[1]}", shape='box', style='filled,rounded',
                       fillcolor='#dbeafe', fontname='SimHei')

    for i in range(N):
        for j in range(N):
            if i != j:
                if adj_matrix[i][j] == 1:
                    dot.edge(str(i), str(j), style='solid', color='#1e293b', penwidth='1.5')
                elif reach_matrix[i][j] == 1 and adj_matrix[i][j] == 0:
                    dot.edge(str(i), str(j), style='dashed', color='#94a3b8', penwidth='1.0')
    st.graphviz_chart(dot, use_container_width=True)

with col_vis2:
    st.subheader("输出 5：MICMAC 分类散点图")
    st.success("四象限图：将因素分为驱动型、依赖型、联系型和自治型。")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = []
    mid_x, mid_y = np.median(dep_pwr), np.median(drive_pwr)
    for i in range(N):
        if dep_pwr[i] <= mid_x and drive_pwr[i] > mid_y:
            colors.append('#dc2626')  # 驱动
        elif dep_pwr[i] > mid_x and drive_pwr[i] <= mid_y:
            colors.append('#2563eb')  # 依赖
        elif dep_pwr[i] > mid_x and drive_pwr[i] > mid_y:
            colors.append('#ea580c')  # 联系
        else:
            colors.append('#16a34a')  # 自治

    ax.scatter(dep_pwr, drive_pwr, c=colors, s=150, alpha=0.8, edgecolors='black')
    for i in range(N): ax.annotate(f"F{i + 1}", (dep_pwr[i] + 0.2, drive_pwr[i] + 0.2), fontsize=11, fontweight='bold')

    ax.axhline(y=mid_y, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=mid_x, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('依赖度 (Dependence Power)', fontsize=12)
    ax.set_ylabel('驱动力 (Driving Power)', fontsize=12)

    ax.text(np.min(dep_pwr), np.max(drive_pwr) - 0.5, '驱动簇 (根源)', color='#dc2626', fontsize=12, fontweight='bold')
    ax.text(np.max(dep_pwr) - 3, np.max(drive_pwr) - 0.5, '联系簇', color='#ea580c', fontsize=12, fontweight='bold')
    ax.text(np.min(dep_pwr), np.min(drive_pwr) + 0.5, '自治簇', color='#16a34a', fontsize=12, fontweight='bold')
    ax.text(np.max(dep_pwr) - 3, np.min(drive_pwr) + 0.5, '依赖簇 (表象)', color='#2563eb', fontsize=12,
            fontweight='bold')

    st.pyplot(fig)

st.markdown("---")
st.subheader("输出 6：关系解释表 (Interpretive Matrix)")
st.info("详细列出带有业务解释的关系矩阵。不仅包含专家的直接经验认知，还包含了算法计算出的“间接连带效应”。")

logic_data = []
for i in range(N):
    for j in range(N):
        if adj_matrix[i][j] == 1 and i != j:
            reason = default_logic.get((i, j), f"业务实证表明，{FACTORS[i]} 显著正向影响/制约了 {FACTORS[j]}。")
            logic_data.append({
                "逻辑流向 (Path)": f"【F{i + 1}】➔【F{j + 1}】",
                "关系性质": "🔍 直接驱动 (Direct)",
                "业务解释说明 (Interpretive Logic)": reason
            })

displayed_trans = 0
for link in transitive_links:
    i, j, k = link
    if abs(i - j) > 3 and displayed_trans < 15:
        logic_data.append({
            "逻辑流向 (Path)": f"【F{i + 1}】➔【F{j + 1}】",
            "关系性质": "🔗 隐性传递 (Transitive)",
            "业务解释说明 (Interpretive Logic)": f"【模型算法深挖】：虽然无直观业务动作联系，但【{FACTORS[i]}】通过中介影响了【{FACTORS[k]}】，进而产生了多米诺骨牌效应，间接制约了【{FACTORS[j]}】。"
        })
        displayed_trans += 1

df_logic = pd.DataFrame(logic_data)
st.dataframe(df_logic, use_container_width=True, height=600)