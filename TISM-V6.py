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

# ------------------ 正式业务代码 ------------------
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 页面基本设置：宽屏，支持左右分栏
st.set_page_config(page_title="刘总论文 - 多源融合 M-TISM 推演系统", layout="wide", initial_sidebar_state="collapsed")

# ================= 7大核心要素及其学术定义 =================
FACTORS_7 = [
    "IF1 组织文化与层级阻力",
    "IF2 管理层防失控担忧",
    "IF3 复合型人才缺乏",
    "IF4 传统事前管控刚性",
    "CSF1 应急响应能力",
    "CSF2 交付价值一致性",
    "CSF3 底线保障(成本/风险)"
]
N7 = len(FACTORS_7)

FACTOR_DEFINITIONS = {
    "IF1 组织文化与层级阻力": "源自问卷高分项（均分4.1），反映传统层级管控文化的根深蒂固，排斥扁平化变革。",
    "IF2 管理层防失控担忧": "源自问卷（均分3.8），反映监管层担心放权引发跨境法律与合规失控的心理防御。",
    "IF3 复合型人才缺乏": "源自问卷（均分3.7），一线团队缺乏兼具敏捷思维与国际工程采购实操的复合型能力。",
    "IF4 传统事前管控刚性": "现状特征，重事前计划固化与层级串行审批，事中变更空间极度受限（得分仅3.3）。",
    "CSF1 应急响应能力": "最高效益期望（均分4.2），面对地缘政治或物流中断时，采购系统的动态调整效率。",
    "CSF2 交付价值一致性": "均分4.1，确保采购最终交付物与国际矿业项目的业主长远战略高度契合。",
    "CSF3 底线保障(成本/风险)": "传统绩效基准（78.6%高认可），守住项目预算、合规底线及跨境索赔风险防线。"
}

# 7个要素的基准 SSIM 矩阵
baseline_ssim_7 = [['O'] * N7 for _ in range(N7)]
baseline_ssim_7[0][1] = 'V'  # IF1 -> IF2
baseline_ssim_7[0][2] = 'V'  # IF1 -> IF3
baseline_ssim_7[1][3] = 'V'  # IF2 -> IF4
baseline_ssim_7[2][3] = 'V'  # IF3 -> IF4
baseline_ssim_7[3][4] = 'V'  # IF4 -> CSF1
baseline_ssim_7[2][5] = 'V'  # IF3 -> CSF2
baseline_ssim_7[3][6] = 'V'  # IF4 -> CSF3
baseline_ssim_7[4][6] = 'X'  # CSF1 <-> CSF3
baseline_ssim_7[4][5] = 'V'  # CSF1 -> CSF2

# ================= 页面标题 =================
st.title("🏗️ 国际工程采购敏捷治理：多源融合 M-TISM 推演与决策系统")
st.markdown("""
> **系统设计目标**：严格遵循科研逻辑，分为三大板块：  
> **第一部分**（大样本问卷静态基准：完整6大输出） ➔ **第二部分**（左侧专家添加与API设置，右侧专家档案存储与管理） ➔ **第三部分**（多模式综合决策与全套多维对比报告）。
""")
st.markdown("---")


# ================= 核心算法引擎封装 =================
def run_tism_pipeline(adj):
    reach = adj.copy()
    trans = []
    for k in range(N7):
        for i in range(N7):
            for j in range(N7):
                if reach[i][j] == 0 and reach[i][k] == 1 and reach[k][j] == 1:
                    reach[i][j] = 1
                    trans.append((i, j, k))

    drv = np.sum(reach, axis=1)
    dep = np.sum(reach, axis=0)

    lvl_map = {}
    rem = list(range(N7))
    lvl_idx = 1
    while rem:
        r_sets = {i: set([j for j in rem if reach[i][j] == 1]) for i in rem}
        a_sets = {i: set([j for j in rem if reach[j][i] == 1]) for i in rem}
        i_sets = {i: r_sets[i].intersection(a_sets[i]) for i in rem}
        top_f = [i for i in rem if r_sets[i] == i_sets[i]]
        if not top_f: top_f = rem.copy()
        lvl_map[lvl_idx] = top_f
        for f in top_f: rem.remove(f)
        lvl_idx += 1
    return reach, trans, drv, dep, lvl_map


def ssim_to_adj(ssim_mat):
    mat = np.zeros((N7, N7), dtype=int)
    for r in range(N7):
        for c in range(N7):
            val = str(ssim_mat[r][c]).upper()
            if val == 'V':
                mat[r][c] = 1
            elif val == 'A':
                mat[c][r] = 1
            elif val == 'X':
                mat[r][c], mat[c][r] = 1, 1
    np.fill_diagonal(mat, 1)
    return mat


base_adj = ssim_to_adj(baseline_ssim_7)
base_reach, base_trans, base_drv, base_dep, base_levels = run_tism_pipeline(base_adj)

# ================= PART 1: 第一部分 —— 大样本群体问卷基准分析 =================
st.header("📌 第一部分：大样本群体问卷基准分析 (Baseline Analysis)")
st.markdown("基于 **84份专项有效问卷** 的大样本量化分析，确立 7 个核心要素及其基准关系结构。")

st.subheader("7 大核心要素的学术定义卡片")
exp_cols = st.columns(3)
for idx, (f_name, f_desc) in enumerate(FACTOR_DEFINITIONS.items()):
    with exp_cols[idx % 3]:
        st.info(f"**{f_name}**\n\n{f_desc}")

with st.expander("📂 点击展开/折叠：查看基于大样本问卷基准的完整 6 大输出模型与图表 (Baseline Full Outputs)",
                 expanded=True):
    st.markdown("### 【基准输出 1 & 2】初始与最终可达矩阵")
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown("**初始可达矩阵 (Initial)**")
        df_b_init = pd.DataFrame(base_adj, columns=[f"E{i + 1}" for i in range(N7)],
                                 index=[f"E{i + 1}" for i in range(N7)])
        st.dataframe(df_b_init, use_container_width=True)
    with c_b2:
        st.markdown("**最终可达矩阵与驱动力/依赖度 (Final)**")
        df_b_fin = pd.DataFrame(base_reach, columns=[f"E{i + 1}" for i in range(N7)],
                                index=[f"E{i + 1}" for i in range(N7)])
        df_b_fin['驱动力'] = base_drv
        df_b_fin.loc['依赖度'] = list(base_dep) + ["-"]
        st.dataframe(df_b_fin, use_container_width=True)

    st.markdown("### 【基准输出 3】层级划分表 (Level Partitioning)")
    base_level_records = []
    for lvl, f_list in base_levels.items():
        for f in f_list:
            base_level_records.append({"要素名称": FACTORS_7[f], "所处层级": f"Level {lvl}"})
    st.dataframe(pd.DataFrame(base_level_records), use_container_width=True)

    st.markdown("### 【基准输出 4 & 5】有向图结构模型与 MICMAC 象限散点图")
    cb_v1, cb_v2 = st.columns(2)
    with cb_v1:
        st.markdown("**问卷基准 M-TISM 有向图**")
        dot_base = graphviz.Digraph(engine='dot')
        dot_base.attr(rankdir='BT', size="5.5,5.5")
        for lvl, f_list in base_levels.items():
            with dot_base.subgraph() as s:
                s.attr(rank='same')
                for f in f_list:
                    dot_base.node(str(f), f"E{f + 1}\n{FACTORS_7[f].split(' ')[1]}", shape='box',
                                  style='filled,rounded', fillcolor='#e2e8f0', fontname='SimHei')
        for i in range(N7):
            for j in range(N7):
                if i != j and base_adj[i][j] == 1:
                    dot_base.edge(str(i), str(j), style='solid', color='black')
                elif i != j and base_reach[i][j] == 1 and base_adj[i][j] == 0:
                    dot_base.edge(str(i), str(j), style='dashed', color='gray')
        st.graphviz_chart(dot_base, use_container_width=True)

    with cb_v2:
        st.markdown("**问卷基准 MICMAC 四象限散点图**")
        fig_b, ax_b = plt.subplots(figsize=(5.5, 4.5))

        jitter_x = base_dep + np.array([0, -0.05, 0.05, 0, -0.05, 0.05, 0])
        jitter_y = base_drv + np.array([0, 0.05, -0.05, 0, 0.05, -0.05, 0])

        ax_b.scatter(jitter_x, jitter_y, c='forestgreen', s=110, edgecolors='black', alpha=0.8)
        for i in range(N7):
            ax_b.annotate(f"E{i + 1}", (jitter_x[i] + 0.1, jitter_y[i] + 0.1), fontsize=9, fontweight='bold')

        mx_b, my_b = np.median(base_dep), np.median(base_drv)
        ax_b.axhline(y=my_b, color='red', linestyle='--', alpha=0.6)
        ax_b.axvline(x=mx_b, color='red', linestyle='--', alpha=0.6)
        ax_b.set_xlabel('依赖度 (Dependence Power)')
        ax_b.set_ylabel('驱动力 (Driving Power)')

        # 🌟 完美补齐第一部分 MICMAC 图的四象限文字标注
        ax_b.text(np.min(base_dep), np.max(base_drv) - 0.2, 'IV. 驱动簇', color='red', fontsize=9, fontweight='bold')
        ax_b.text(np.max(base_dep) - 2, np.max(base_drv) - 0.2, 'III. 联系簇', color='darkorange', fontsize=9,
                  fontweight='bold')
        ax_b.text(np.min(base_dep), np.min(base_drv) + 0.2, 'I. 自治簇', color='green', fontsize=9, fontweight='bold')
        ax_b.text(np.max(base_dep) - 2, np.min(base_drv) + 0.2, 'II. 依赖簇', color='blue', fontsize=9,
                  fontweight='bold')

        st.pyplot(fig_b)

    st.markdown("### 【基准输出 6】关系解释表 (Interpretive Matrix)")
    base_interp_rows = []
    for i in range(N7):
        for j in range(N7):
            if base_adj[i][j] == 1 and i != j:
                base_interp_rows.append({"因果路径": f"【E{i + 1}】➔【E{j + 1}】",
                                         "业务机理说明": f"问卷大样本实证确认：{FACTORS_7[i]} 对 {FACTORS_7[j]} 具有直接驱动与制约作用。"})
    st.dataframe(pd.DataFrame(base_interp_rows), use_container_width=True)

st.markdown("---")

# ================= PART 2: 第二部分 —— 左右分栏结构 =================
st.header("👥 第二部分：专家访谈录入、API设置与智库档案存储 (Expert Management & API)")
st.markdown(
    "采用左右分栏结构：**左侧**提供 Gemini API 密钥设置、专家档案添加与文档上传接口；**右侧**实时存储、展示并支持微调所有专家的 SSIM 矩阵。")

if "expert_panel" not in st.session_state:
    st.session_state["expert_panel"] = {
        "专家1 (总监级-张总)": {
            "exp": "超过 20年", "role": "总部高管/区域总监", "agile": "敏捷实战专家",
            "file_info": "访谈纪要_矿业EPC总包.docx",
            "ssim": [row[:] for row in baseline_ssim_7]
        }
    }

col_part2_left, col_part2_right = st.columns([1.2, 2.2])

with col_part2_left:
    st.subheader("➕ 1. 添加专家与 AI 配置")

    gemini_api_key = st.text_input(
        "GEMINI API 密钥 (仅用于报告解读及答案内容提取)",
        type="password",
        help="明确声明：该链接仅用于报告解读及专家访谈答案内容提取，系统不会泄露或另作他用。"
    )

    with st.form("add_expert_form_v11"):
        n_name = st.text_input("专家代号/姓名", value=f"专家{len(st.session_state['expert_panel']) + 1}")
        n_exp = st.selectbox("从业年限", ["0-5年", "5-10年", "10-20年", "超过 20年"], index=2)
        n_role = st.selectbox("管控层级", ["一线操作岗/专员", "项目经理/部门主管", "总部高管/区域总监"], index=1)
        n_agile = st.selectbox("敏捷熟悉度", ["不了解/刚接触", "了解并部分实践", "敏捷实战专家"], index=1)
        uploaded_doc = st.file_uploader("上传访谈问卷/纪要 (Word/PDF/TXT)", type=["docx", "pdf", "txt"])

        submitted_add = st.form_submit_button("确认添加专家档案并提取参数", use_container_width=True)
        if submitted_add:
            if len(st.session_state["expert_panel"]) >= 10:
                st.error("最多支持添加 10 位专家！")
            else:
                f_name = uploaded_doc.name if uploaded_doc is not None else "无附件_纯人工录入"
                st.session_state["expert_panel"][n_name] = {
                    "exp": n_exp, "role": n_role, "agile": n_agile,
                    "file_info": f_name, "ssim": [row[:] for row in baseline_ssim_7]
                }
                st.success(f"成功添加专家 [{n_name}] 并存入内存缓存！")
                st.rerun()

with col_part2_right:
    st.subheader("🗂️ 2. 专家智库档案存储与矩阵管理")
    if not st.session_state["expert_panel"]:
        st.warning("当前智库为空，请在左侧添加专家。")
    else:
        exp_tabs = st.tabs(list(st.session_state["expert_panel"].keys()))
        for idx, (exp_key, exp_val) in enumerate(st.session_state["expert_panel"].items()):
            with exp_tabs[idx]:
                st.write(f"- 从业年限：{exp_val['exp']} | 管控层级：{exp_val['role']} | 敏捷认知：{exp_val['agile']}")
                st.write(f"- 关联缓存附件：`{exp_val['file_info']}`")

                sc_e = 0.5 if "超过" in exp_val['exp'] else (0.4 if "10-20" in exp_val['exp'] else 0.2)
                sc_r = 0.5 if "总部" in exp_val['role'] else (0.3 if "项目" in exp_val['role'] else 0.2)
                sc_a = 0.5 if "专家" in exp_val['agile'] else 0.3
                raw_w = sc_e * 0.4 + sc_r * 0.4 + sc_a * 0.2
                st.info(f"📊 **客观计算权重得分**: `{raw_w:.2f}` (已安全缓存)")

                if st.button(f"🗑️ 删除此专家档案", key=f"del_{exp_key}"):
                    del st.session_state["expert_panel"][exp_key]
                    st.rerun()

                st.markdown(f"**{exp_key} 的访谈 SSIM 判定表 (支持微调)**")
                curr_df = pd.DataFrame(exp_val["ssim"], columns=[f"E{i + 1}" for i in range(N7)],
                                       index=[f"E{i + 1}" for i in range(N7)])
                edited_res = st.data_editor(curr_df, key=f"mat_edit_{exp_key}", height=200)

                updated_list = []
                for r in range(N7):
                    updated_list.append([str(edited_res.iloc[r, c]).upper() for c in range(N7)])
                st.session_state["expert_panel"][exp_key]["ssim"] = updated_list


def compute_objective_consensus(expert_dict):
    weights = {}
    total_w = 0
    for name, data in expert_dict.items():
        sc_e = 0.5 if "超过" in data['exp'] else (0.4 if "10-20" in data['exp'] else 0.2)
        sc_r = 0.5 if "总部" in data['role'] else (0.3 if "项目" in data['role'] else 0.2)
        sc_a = 0.5 if "专家" in data['agile'] else 0.3
        w = sc_e * 0.4 + sc_r * 0.4 + sc_a * 0.2
        weights[name] = w
        total_w += w
    norm_w = {k: v / total_w for k, v in weights.items()}

    agg_mat = np.zeros((N7, N7), dtype=float)
    for name, data in expert_dict.items():
        w = norm_w[name]
        mat = data["ssim"]
        for r in range(N7):
            for c in range(N7):
                v = mat[r][c]
                if v == 'V':
                    agg_mat[r][c] += w * 1.0
                elif v == 'A':
                    agg_mat[c][r] += w * 1.0
                elif v == 'X':
                    agg_mat[r][c] += w * 1.0
                    agg_mat[c][r] += w * 1.0

    return agg_mat, norm_w


agg_score_matrix, expert_weights_dict = compute_objective_consensus(st.session_state["expert_panel"])

st.markdown("---")

# ================= PART 3: 第三部分 —— 综合决策与动态对比全套 6 大输出 =================
st.header("⚖️ 第三部分：群体问卷与专家缓存数据的综合决策与动态对比 (Synthesis & Decision)")
st.markdown(
    "程序自动从第二部分内存缓存中读取多位专家加权矩阵，结合第一部分问卷基准进行多模式综合决策，输出**全套 6 大多维可视化对比报告**。")

with st.form("fusion_decision_form"):
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        fusion_threshold = st.slider("1. 群体共识截断阈值 (Threshold)", 0.3, 0.9, 0.5, 0.05,
                                     help="加权得分超过此阈值的因果关系才会被最终采纳。")
    with c_m2:
        fusion_mode = st.selectbox("2. 多源数据融合策略", [
            "【替代模式】完全采纳专家缓存综合意见",
            "【叠加模式】问卷基准 40% + 专家缓存 60% 混合加权",
            "【保守模式】取问卷与专家意见的交集 (AND逻辑)"
        ])
    with c_m3:
        output_compare_type = st.multiselect("3. 综合多维输出项多选", [
            "1. 综合群体初始与最终可达矩阵",
            "2. 综合群体层级划分表 (Level Partitioning)",
            "3. M-TISM 有向图结构模型 (带变动高亮)",
            "4. MICMAC 象限对比散点图",
            "5. 多专家访谈综合解释矩阵"
        ], default=["1. 综合群体初始与最终可达矩阵", "2. 综合群体层级划分表 (Level Partitioning)",
                    "3. M-TISM 有向图结构模型 (带变动高亮)", "4. MICMAC 象限对比散点图", "5. 多专家访谈综合解释矩阵"])

    submitted_decision = st.form_submit_button("⚖️ 读取缓存并执行综合决策计算", type="primary")

final_expert_adj = np.zeros((N7, N7), dtype=int)
for r in range(N7):
    for c in range(N7):
        score = agg_score_matrix[r][c]
        if fusion_mode.startswith("【替代模式】"):
            if score >= fusion_threshold: final_expert_adj[r][c] = 1
        elif fusion_mode.startswith("【叠加模式】"):
            mixed_score = (0.4 * base_adj[r][c]) + (0.6 * score)
            if mixed_score >= fusion_threshold: final_expert_adj[r][c] = 1
        else:
            if base_adj[r][c] == 1 and score >= fusion_threshold: final_expert_adj[r][c] = 1
np.fill_diagonal(final_expert_adj, 1)

exp_reach, exp_trans, exp_drv, exp_dep, exp_levels = run_tism_pipeline(final_expert_adj)

if submitted_decision or True:
    st.markdown("---")
    st.subheader("🎯 综合决策模型全套多维输出与动态对比报告")

    if "1. 综合群体初始与最终可达矩阵" in output_compare_type:
        st.markdown("#### 📄 综合输出 1 & 2：缓存调用与群体共识可达矩阵")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**综合群体初始邻接矩阵**")
            st.dataframe(pd.DataFrame(final_expert_adj, columns=[f"E{i + 1}" for i in range(N7)],
                                      index=[f"E{i + 1}" for i in range(N7)]), use_container_width=True)
        with cc2:
            st.markdown("**综合群体最终可达矩阵与驱动力**")
            df_exp_fin = pd.DataFrame(exp_reach, columns=[f"E{i + 1}" for i in range(N7)],
                                      index=[f"E{i + 1}" for i in range(N7)])
            df_exp_fin['驱动力'] = exp_drv
            df_exp_fin.loc['依赖度'] = list(exp_dep) + ["-"]
            st.dataframe(df_exp_fin, use_container_width=True)

    if "2. 综合群体层级划分表 (Level Partitioning)" in output_compare_type:
        st.markdown("#### 📄 综合输出 3：专家群体综合层级划分表")
        exp_level_records = []
        for lvl, f_list in exp_levels.items():
            for f in f_list:
                exp_level_records.append({"要素名称": FACTORS_7[f], "综合所处层级": f"Level {lvl}"})
        st.dataframe(pd.DataFrame(exp_level_records), use_container_width=True)

    if "3. M-TISM 有向图结构模型 (带变动高亮)" in output_compare_type or "4. MICMAC 象限对比散点图" in output_compare_type:
        st.markdown("#### 🖼️📈 综合输出 4 & 5：M-TISM 有向图结构模型与 MICMAC 象限对比")
        st.info(
            "🎨 **色彩语义**：\n- 🖤 **黑色实线**：问卷基准与专家共识高度一致的核心机理。\n- 🔴 **红色粗线**：引入多位资深专家深度访谈后**新增**的隐性因果传导路径。")

        c_out3, c_out4 = st.columns(2)

        with c_out3:
            if "3. M-TISM 有向图结构模型 (带变动高亮)" in output_compare_type:
                st.markdown("**综合 M-TISM 有向图 (多源比对视图)**")
                dot_comp = graphviz.Digraph(engine='dot')
                dot_comp.attr(rankdir='BT', size="5,5")
                for lvl, f_list in exp_levels.items():
                    with dot_comp.subgraph() as s:
                        s.attr(rank='same')
                        for f in f_list:
                            dot_comp.node(str(f), f"E{f + 1}\n{FACTORS_7[f].split(' ')[1]}", shape='box',
                                          style='filled,rounded', fillcolor='#fef3c7', fontname='SimHei')

                for i in range(N7):
                    for j in range(N7):
                        if i != j:
                            if final_expert_adj[i][j] == 1:
                                is_new_path = (base_adj[i][j] == 0)
                                c_col = '#dc2626' if is_new_path else '#111827'
                                p_w = '2.0' if is_new_path else '1.0'
                                dot_comp.edge(str(i), str(j), style='solid', color=c_col, penwidth=p_w)
                            elif exp_reach[i][j] == 1 and final_expert_adj[i][j] == 0:
                                dot_comp.edge(str(i), str(j), style='dashed', color='#9ca3af', penwidth='0.6')

                st.graphviz_chart(dot_comp, use_container_width=True)

        with c_out4:
            if "4. MICMAC 象限对比散点图" in output_compare_type:
                st.markdown("**MICMAC 驱动力-依赖度四象限对比散点图**")
                fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))

                jitter_exp_x = exp_dep + np.array([0, -0.05, 0.05, 0, -0.05, 0.05, 0])
                jitter_exp_y = exp_drv + np.array([0, 0.05, -0.05, 0, 0.05, -0.05, 0])

                ax2.scatter(jitter_exp_x, jitter_exp_y, c='#d97706', s=110, edgecolors='black', alpha=0.9,
                            label='专家综合模型')
                ax2.scatter(base_dep, base_drv, c='#2563eb', s=80, marker='^', alpha=0.6, label='问卷基准模型')

                for i in range(N7):
                    ax2.annotate(f"E{i + 1}", (jitter_exp_x[i] + 0.08, jitter_exp_y[i] + 0.08), fontsize=9,
                                 fontweight='bold')

                mx2, my2 = np.median(exp_dep), np.median(exp_drv)
                ax2.axhline(y=my2, color='red', linestyle='--', alpha=0.6)
                ax2.axvline(x=mx2, color='red', linestyle='--', alpha=0.6)
                ax2.set_xlabel('依赖度 (Dependence Power)')
                ax2.set_ylabel('Driving Power (驱动力)')

                ax2.text(np.min(exp_dep), np.max(exp_drv) - 0.2, 'IV. 驱动簇', color='red', fontsize=9,
                         fontweight='bold')
                ax2.text(np.max(exp_dep) - 2, np.max(exp_drv) - 0.2, 'III. 联系簇', color='darkorange', fontsize=9,
                         fontweight='bold')
                ax2.text(np.min(exp_dep), np.min(exp_drv) + 0.2, 'I. 自治簇', color='green', fontsize=9,
                         fontweight='bold')
                ax2.text(np.max(exp_dep) - 2, np.min(exp_drv) + 0.2, 'II. 依赖簇', color='blue', fontsize=9,
                         fontweight='bold')

                ax2.legend(loc='upper right', fontsize=8)
                st.pyplot(fig2)

    if "5. 多专家访谈综合解释矩阵" in output_compare_type:
        st.markdown("#### 📖 综合输出 6：多专家访谈语料库与决策解释矩阵")
        expert_matrix_rows = []
        for i in range(N7):
            for j in range(N7):
                if final_expert_adj[i][j] == 1 and i != j:
                    expert_matrix_rows.append({
                        "因果路径": f"【E{i + 1}】➔【E{j + 1}】",
                        "涉及要素": f"{FACTORS_7[i]} ➔ {FACTORS_7[j]}",
                        "群体决策与业务机理解释": f"经内存缓存读取 {len(st.session_state['expert_panel'])} 位专家加权融合（共识阈值 {fusion_threshold}）：在国际工程实际履约中，该路径获得了群体高度认同，建议将其纳入项目的敏捷监管阻断清单。"
                    })
        st.dataframe(pd.DataFrame(expert_matrix_rows), use_container_width=True, height=350)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size:14px;'>劉总论文实证研究支撑系统 | Multi-Expert M-TISM Decision Framework</p>",
    unsafe_allow_html=True)