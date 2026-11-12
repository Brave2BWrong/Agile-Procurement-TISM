# -*- coding: utf-8 -*-
"""
国际工程采购敏捷治理：多源融合 M-TISM 推演与决策系统  ——  V8.1

运行方式（本机 / Streamlit Cloud 一致）：
    streamlit run TISM-V8.1.py

V8.1 相对 V8 的变更清单（2026-11-12）：
  1. [布局 A] 所有"两张语义不同的表并排"处，改为独立卡片(st.container border=True)
     + 中间空白间隔列，明确视觉边界；卡片内补一句"这张表说的是什么"的说明。
  2. [布局 B] 下载按钮从各列内部移到列外同一行，绝对水平对齐，消除"一高一低"错位。
  3. [布局 C] 两张表设相同高度、组后留白与分隔，避免与下一节粘连。
     实测依据：egolite DOM 测量显示宽屏 1553px / 窄屏 900px 下 st.columns(2)
     间距均恒为 16px、dataframe 撑满整列且无溢出 -> 非浏览器问题，是布局设定问题。
  4. [专家区] 删除整个"添加专家与 AI 配置"左栏：GEMINI API 输入框、添加专家表单、
     附件上传、一键导入 E01/E02、删除专家档案、可微调 SSIM 编辑器，全部移除。
  5. [专家区] 改为两位专家各一个页签，只读展示真实数据：
     数据源信息概要（取自两份访谈问卷 docx 原件）+ 7x7 SSIM 判定表
     + 21 组关系判定明细 + 因果骨架校验要点。
  6. [权重] 展开完整可复算的计算依据链：三维度系数来源与分档标准、逐人打分、
     原始分代入式、归一化公式、聚合规则、以及权重对阈值与成边的实际影响。
  7. [交互] 输出项多选纳入参数指纹；报告头逐项列出"本次生效参数"；
     每个控件补作用说明；全部按钮/下载/清除均真实生效。
  8. 数据源全部固定为模块级常量，结果可复现；版本号统一为 V8.1。

说明：本文件不覆盖 TISM-V8.py / TISM-V7.py，二者作为归档基线保持冻结。
"""

import json
import platform

import numpy as np
import pandas as pd
import graphviz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

# ---------------- 字体（本机 Windows / 云端 Linux 双兼容） ----------------
IS_WINDOWS = platform.system() == "Windows"
GV_FONT = "Microsoft YaHei" if IS_WINDOWS else "WenQuanYi Zen Hei"
MPL_FONTS = (
    ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    if IS_WINDOWS
    else ["WenQuanYi Zen Hei", "DejaVu Sans"]
)
plt.rcParams["font.sans-serif"] = MPL_FONTS
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="刘总论文 - 多源融合 M-TISM 推演系统 V8.1",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================= 7 大核心要素及其学术定义 =================
FACTORS_7 = [
    "IF1 组织文化与层级阻力",
    "IF2 管理层防失控担忧",
    "IF3 复合型人才缺乏",
    "IF4 传统事前管控刚性",
    "CSF1 应急响应能力",
    "CSF2 交付价值一致性",
    "CSF3 底线保障(成本/风险)",
]
N7 = len(FACTORS_7)

FACTOR_DEFINITIONS = {
    "IF1 组织文化与层级阻力": "源自问卷高分项（均分4.1），反映传统层级管控文化的根深蒂固，排斥扁平化变革。",
    "IF2 管理层防失控担忧": "源自问卷（均分3.8），反映监管层担心放权引发跨境法律与合规失控的心理防御。",
    "IF3 复合型人才缺乏": "源自问卷（均分3.7），一线团队缺乏兼具敏捷思维与国际工程采购实操的复合型能力。",
    "IF4 传统事前管控刚性": "现状特征，重事前计划固化与层级串行审批，事中变更空间极度受限（得分仅3.3）。",
    "CSF1 应急响应能力": "最高效益期望（均分4.2），面对地缘政治或物流中断时，采购系统的动态调整效率。",
    "CSF2 交付价值一致性": "均分4.1，确保采购最终交付物与国际矿业项目的业主长远战略高度契合。",
    "CSF3 底线保障(成本/风险)": "传统绩效基准（78.6%高认可），守住项目预算、合规底线及跨境索赔风险防线。",
}

# 问卷基准 SSIM（行 i → 列 j；V/A/X/O）
baseline_ssim_7 = [
    ["O", "V", "V", "O", "O", "O", "O"],
    ["O", "O", "O", "V", "O", "O", "O"],
    ["O", "O", "O", "V", "O", "V", "O"],
    ["O", "O", "O", "O", "V", "O", "V"],
    ["O", "O", "O", "O", "O", "V", "X"],
    ["O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "O"],
]

# 两位真实专家的 SSIM 判定（存档文档 5.4 节 / 两份访谈问卷附录汇总矩阵）
ssim_e01 = [
    ["O", "X", "V", "V", "O", "O", "O"],
    ["O", "O", "X", "V", "V", "O", "O"],
    ["O", "O", "O", "X", "V", "V", "V"],
    ["O", "O", "O", "O", "V", "V", "V"],
    ["O", "O", "O", "O", "O", "V", "V"],
    ["O", "O", "O", "O", "O", "O", "V"],
    ["O", "O", "O", "O", "O", "O", "O"],
]

ssim_e02 = [
    ["O", "X", "V", "V", "V", "V", "O"],
    ["O", "O", "X", "X", "O", "O", "O"],
    ["O", "O", "O", "X", "V", "V", "V"],
    ["O", "O", "O", "O", "V", "V", "V"],
    ["O", "O", "O", "O", "O", "V", "V"],
    ["O", "O", "O", "O", "O", "O", "V"],
    ["O", "O", "O", "O", "O", "O", "O"],
]

# 21 组关系判定明细（存档 5.3 节逐组对照，原文数据）
RELATION_21 = [
    ("M1-01", "IF1 vs IF2", "X", "X", "一致"),
    ("M1-02", "IF1 vs IF3", "V", "V", "一致"),
    ("M1-03", "IF1 vs IF4", "V", "V", "一致"),
    ("M1-04", "IF2 vs IF3", "X", "X", "一致"),
    ("M1-05", "IF2 vs IF4", "V", "X", "分歧：E02 认为双向"),
    ("M1-06", "IF3 vs IF4", "X", "X", "一致"),
    ("M2-01", "IF1 vs CSF1", "O", "V", "分歧：观测层级不同"),
    ("M2-02", "IF1 vs CSF2", "O", "V", "分歧：观测层级不同"),
    ("M2-03", "IF1 vs CSF3", "O", "O", "一致"),
    ("M2-04", "IF2 vs CSF1", "V", "O", "分歧：E01 判负向直接影响"),
    ("M2-05", "IF2 vs CSF2", "O", "O", "一致"),
    ("M2-06", "IF2 vs CSF3", "O", "O", "一致"),
    ("M2-07", "IF3 vs CSF1", "V", "V", "一致"),
    ("M2-08", "IF3 vs CSF2", "V", "V", "一致"),
    ("M2-09", "IF3 vs CSF3", "V", "V", "一致"),
    ("M2-10", "IF4 vs CSF1", "V", "V", "一致"),
    ("M2-11", "IF4 vs CSF2", "V", "V", "一致"),
    ("M2-12", "IF4 vs CSF3", "V", "V", "一致"),
    ("M3-01", "CSF1 vs CSF2", "V", "V", "一致"),
    ("M3-02", "CSF1 vs CSF3", "V", "V", "一致"),
    ("M3-03", "CSF2 vs CSF3", "V", "V", "一致"),
]

# 两位专家的档案（全部取自两份访谈问卷 docx 原件首页与附录）
EXPERT_DOSSIER = {
    "E01": {
        "tab": "E01 · EPC 总承包执行侧",
        "name": "E01 央企海外矿业EPC总承包单位副总",
        "code": "SIM-E01",
        "perspective": "执行侧 / 合同交付视角",
        "role": "央企二级单位分管国际工程副总经理｜海外矿业EPC总承包单位",
        "duty": "总承包合同交付、工期、项目利润、回款、业主验收与索赔控制",
        "boundary": "集团制度、审计和考核约束；集团通常不参与项目逐项审批",
        "scope": "主要基于EPC建设交付，对投产后的长期运营保产采购关注较少（视角限制）",
        "date": "2026年11月1日",
        "source": "基于敏捷治理的国际工程采购监管 TISM 专家深度访谈问卷_专家01_央企海外矿业EPC总承包单位副总.docx",
        "counts": "V=13 · A=0 · X=3 · O=5（共 21 组，仅填上三角）",
        "skeleton": [
            ("最底层根源障碍", "IF1 组织文化与层级阻力、IF2 管理层防失控担忧、IF3 复合型人才缺乏"),
            ("关键中介/能力节点", "IF4 传统事前管控与合规刚性（制度中介）与 IF3（能力节点）"),
            ("优先干预路径", "IF2 → IF4 → CSF1"),
            ("主要恶性循环", "IF1↔IF2；IF2↔IF3；IF3↔IF4（以 IF3→IF4 为主）"),
            ("最小可行试点", "失能分包商紧急替换：以可逆准备与不可逆承诺分离为核心"),
            ("价值边界", "关键工期优先，其次项目利润与回款；安全、主工艺性能、法律廉洁与最终验收为不可交易红线"),
        ],
        "ssim": ssim_e01,
        "exp": "超过 20年",
        "role_level": "项目经理/部门主管",
        "agile": "敏捷实战专家",
        "e_score": 0.5,
        "r_score": 0.3,
        "a_score": 0.5,
        "raw": 0.42,
        "norm": 0.4565,
    },
    "E02": {
        "tab": "E02 · 总部采购监管侧",
        "name": "E02 央企总部采购供应链监管部门主管",
        "code": "SIM-E02",
        "perspective": "监管侧 / 制度设计视角",
        "role": "国资直属央企总部采购供应链监管部门主管",
        "duty": "采购制度建设、专项审查、抽查审计、整改跟踪和穿透式监督",
        "boundary": "不具有具体采购事项的审批或者否决权；不持续跟踪精细运营采购数据",
        "scope": "二级单位及所属国内外矿山的新建、扩建和正常运营期采购（监督范围）",
        "date": "2026年11月1日",
        "source": "基于敏捷治理的国际工程采购监管 TISM 专家深度访谈问卷_专家02_央企总部采购供应链监管部门主管.docx",
        "counts": "V=13 · A=0 · X=4 · O=4（共 21 组，仅填上三角）",
        "skeleton": [
            ("最底层根源障碍", "IF1 组织文化与层级阻力、IF2 管理层防失控担忧、IF3 复合型人才缺乏"),
            ("关键中介/能力节点", "IF4 表现为总部统一底线与二级单位属地细则之间的制度接口缺口"),
            ("优先干预路径", "IF2 ↔ IF4 → CSF1"),
            ("主要恶性循环", "统一制度不适配→例外与备案增加→留痕质量下降→总部信任下降→更不愿放权→IF4继续强化"),
            ("最小可行试点", "海外矿山高频偏离事项数字化事中提醒；提醒不作为审批节点"),
            ("价值边界", "在不降低法律、廉洁和集团底线前提下，同时观察生产连续性、重复异常与制度适配效果"),
        ],
        "ssim": ssim_e02,
        "exp": "超过 20年",
        "role_level": "总部高管/区域总监",
        "agile": "敏捷实战专家",
        "e_score": 0.5,
        "r_score": 0.5,
        "a_score": 0.5,
        "raw": 0.50,
        "norm": 0.5435,
    },
}

# 固定专家面板：不再提供在线编辑，结果可复现
EXPERT_PANEL = {
    d["name"]: {
        "exp": d["exp"],
        "role": d["role_level"],
        "agile": d["agile"],
        "ssim": d["ssim"],
    }
    for d in EXPERT_DOSSIER.values()
}

SSIM_LEGEND = (
    "行 i → 列 j。V = 行因素直接影响列因素；A = 列因素直接影响行因素；"
    "X = 双向直接影响；O = 无第一层直接显著关系；对角线为自反，不计入 21 组判定。"
)


def short_label(idx):
    raw = FACTORS_7[idx].split(" ")[1]
    raw = raw.split("(")[0].split("（")[0]
    return raw[:4]


def make_reach_df(reach, drv, dep):
    """「驱动力」列统一为字符串，避免 '-' 混入整型列触发 pyarrow ArrowInvalid。"""
    labels = [f"E{i + 1}" for i in range(N7)]
    df = pd.DataFrame(reach.astype(object), columns=labels, index=labels)
    df["驱动力"] = [str(int(x)) for x in drv]
    df.loc["依赖度"] = [int(x) for x in dep] + ["-"]
    return df


def ssim_df(mat):
    return pd.DataFrame(
        mat,
        columns=[f"E{i + 1}" for i in range(N7)],
        index=[f"E{i + 1}" for i in range(N7)],
    )


def level_records(levels, col_name):
    return [
        {"要素名称": FACTORS_7[f], col_name: f"Level {lvl}"}
        for lvl, f_list in levels.items()
        for f in f_list
    ]


def download_csv(df, filename, label):
    st.download_button(
        label=label,
        data=df.to_csv(index=True).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


# ================= 布局：两张语义不同的表 =================
PAIR_SPEC = [1, 0.05, 1]          # 中间列专作间隔
TABLE_HEIGHT = 340                # 两表等高，消除高度差


def pair_columns():
    """返回 (左列, 右列)，中间固定留一列空白作为分隔。"""
    left, _gap, right = st.columns(PAIR_SPEC, gap="small")
    return left, right


def spacer(px=14):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def paired_tables(
    title_a, note_a, df_a, file_a, btn_a,
    title_b, note_b, df_b, file_b, btn_b,
):
    """两张不同的表：各自独立卡片 + 中间间隔 + 下载按钮外置同一行绝对对齐。"""
    ca, cb = pair_columns()
    with ca:
        with st.container(border=True):
            st.markdown(f"**{title_a}**")
            st.caption(note_a)
            st.dataframe(df_a, width="stretch", height=TABLE_HEIGHT)
    with cb:
        with st.container(border=True):
            st.markdown(f"**{title_b}**")
            st.caption(note_b)
            st.dataframe(df_b, width="stretch", height=TABLE_HEIGHT)

    spacer(8)
    da, db = pair_columns()
    with da:
        download_csv(df_a, file_a, btn_a)
    with db:
        download_csv(df_b, file_b, btn_b)
    spacer(24)


# ================= 核心算法引擎 =================
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
        if not top_f:
            top_f = rem.copy()
        lvl_map[lvl_idx] = top_f
        for f in top_f:
            rem.remove(f)
        lvl_idx += 1
    return reach, trans, drv, dep, lvl_map


def ssim_to_adj(ssim_mat):
    mat = np.zeros((N7, N7), dtype=int)
    for r in range(N7):
        for c in range(N7):
            val = str(ssim_mat[r][c]).upper()
            if val == "V":
                mat[r][c] = 1
            elif val == "A":
                mat[c][r] = 1
            elif val == "X":
                mat[r][c], mat[c][r] = 1, 1
    np.fill_diagonal(mat, 1)
    return mat


def compute_objective_consensus(expert_dict):
    """年限 0.4 + 层级 0.4 + 敏捷 0.2，归一化后加权聚合。"""
    if not expert_dict:
        return np.zeros((N7, N7), dtype=float), {}
    weights = {}
    total_w = 0.0
    for name, data in expert_dict.items():
        sc_e = 0.5 if "超过" in data["exp"] else (0.4 if "10-20" in data["exp"] else 0.2)
        sc_r = 0.5 if "总部" in data["role"] else (0.3 if "项目" in data["role"] else 0.2)
        sc_a = 0.5 if "专家" in data["agile"] else 0.3
        w = sc_e * 0.4 + sc_r * 0.4 + sc_a * 0.2
        weights[name] = w
        total_w += w
    if total_w <= 0:
        return np.zeros((N7, N7), dtype=float), {}
    norm_w = {k: v / total_w for k, v in weights.items()}

    agg_mat = np.zeros((N7, N7), dtype=float)
    for name, data in expert_dict.items():
        w = norm_w[name]
        mat = data["ssim"]
        for r in range(N7):
            for c in range(N7):
                v = str(mat[r][c]).upper()
                if v == "V":
                    agg_mat[r][c] += w
                elif v == "A":
                    agg_mat[c][r] += w
                elif v == "X":
                    agg_mat[r][c] += w
                    agg_mat[c][r] += w
    return agg_mat, norm_w


def build_fused_adj(agg, base, mode, threshold):
    out = np.zeros((N7, N7), dtype=int)
    for r in range(N7):
        for c in range(N7):
            score = agg[r][c]
            if mode.startswith("【替代模式】"):
                if score >= threshold:
                    out[r][c] = 1
            elif mode.startswith("【叠加模式】"):
                if (0.4 * base[r][c]) + (0.6 * score) >= threshold:
                    out[r][c] = 1
            else:
                if base[r][c] == 1 and score >= threshold:
                    out[r][c] = 1
    np.fill_diagonal(out, 1)
    return out


def decision_fingerprint(threshold, mode, outputs):
    """输出项多选也纳入指纹：改动后未重新执行会明确提示。"""
    return json.dumps(
        {
            "threshold": float(threshold),
            "mode": str(mode),
            "outputs": sorted(list(outputs)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


# ================= 可视化 =================
def draw_digraph(levels, adj, reach, fillcolor, base_adj=None):
    dot = graphviz.Digraph(engine="dot")
    dot.attr(rankdir="BT", size="6,6", nodesep="0.6", ranksep="0.8")
    for lvl, f_list in levels.items():
        with dot.subgraph() as s:
            s.attr(rank="same")
            for f in f_list:
                dot.node(
                    str(f), f"E{f + 1} {short_label(f)}", shape="box",
                    style="filled,rounded", fillcolor=fillcolor,
                    fontname=GV_FONT, fontsize="10", margin="0.16,0.10",
                )
    for i in range(N7):
        for j in range(N7):
            if i == j:
                continue
            if adj[i][j] == 1:
                is_new = base_adj is not None and base_adj[i][j] == 0
                dot.edge(
                    str(i), str(j), style="solid",
                    color="#dc2626" if is_new else "#111827",
                    penwidth="2.0" if is_new else "1.0",
                )
            elif reach[i][j] == 1:
                dot.edge(str(i), str(j), style="dashed", color="#9ca3af", penwidth="0.6")
    return dot


def _quadrant_label(ax, x, y, text, color, va, ha):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=9, color=color,
        va=va, ha=ha, zorder=6,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )


def draw_micmac(dep, drv, main_label, base_dep=None, base_drv=None):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    jitter = np.array([0, -0.05, 0.05, 0, -0.05, 0.05, 0])
    jx = np.asarray(dep, dtype=float) + jitter
    jy = np.asarray(drv, dtype=float) + jitter

    if base_dep is not None:
        ax.scatter(base_dep, base_drv, c="#2563eb", s=70, marker="^", alpha=0.65,
                   label="问卷基准模型", zorder=2)
    ax.scatter(jx, jy, c="#d97706" if base_dep is not None else "forestgreen",
               s=100, edgecolors="black", alpha=0.9, label=main_label, zorder=3)
    for i in range(N7):
        ax.annotate(f"E{i + 1}", (jx[i] + 0.08, jy[i] + 0.08), fontsize=9, zorder=4)

    ax.axhline(y=float(np.median(drv)), color="#dc2626", linestyle="--", alpha=0.6)
    ax.axvline(x=float(np.median(dep)), color="#dc2626", linestyle="--", alpha=0.6)
    ax.set_xlabel("依赖度 (Dependence Power)")
    ax.set_ylabel("驱动力 (Driving Power)")

    _quadrant_label(ax, 0.02, 0.97, "IV. 驱动簇 (根源)", "red", "top", "left")
    _quadrant_label(ax, 0.98, 0.97, "III. 联系簇", "darkorange", "top", "right")
    _quadrant_label(ax, 0.02, 0.03, "I. 自治簇", "green", "bottom", "left")
    _quadrant_label(ax, 0.98, 0.03, "II. 依赖簇 (表象)", "blue", "bottom", "right")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.26, top=0.93)
    return fig


NODE_LEGEND = (
    "节点编号：E1 组织文化 · E2 防失控 · E3 人才 · E4 事前管控 · "
    "E5 应急响应 · E6 交付价值 · E7 底线保障"
)

base_adj = ssim_to_adj(baseline_ssim_7)
base_reach, base_trans, base_drv, base_dep, base_levels = run_tism_pipeline(base_adj)

# ================= 页面标题 =================
st.title("🏗️ 国际工程采购敏捷治理：多源融合 M-TISM 推演与决策系统")
st.markdown(
    "> **V8.1**：两张语义不同的矩阵已各自独立成卡片、留出间距并对齐下载按钮；"
    "第二部分改为两位专家的真实访谈数据档案（只读），并展开权重的完整计算依据；"
    "第三部分全部控件与按钮真实生效，结果按本次执行参数冻结。"
)
st.markdown(
    "> **系统结构**：**第一部分**（84 份问卷静态基准，完整 6 大输出） ➔ "
    "**第二部分**（E01/E02 真实访谈数据源与 SSIM 判定档案、权重计算依据） ➔ "
    "**第三部分**（多模式综合决策与全套多维对比报告）。"
)
st.markdown("---")

# ================= PART 1 =================
st.header("📌 第一部分：大样本群体问卷基准分析 (Baseline Analysis)")
st.markdown("基于 **84 份专项有效问卷** 的大样本量化分析，确立 7 个核心要素及其基准关系结构。")

st.subheader("7 大核心要素的学术定义卡片")
exp_cols = st.columns(3)
for idx, (f_name, f_desc) in enumerate(FACTOR_DEFINITIONS.items()):
    with exp_cols[idx % 3]:
        st.info(f"**{f_name}**\n\n{f_desc}")

with st.expander(
    "📂 点击展开/折叠：查看基于大样本问卷基准的完整 6 大输出模型与图表 (Baseline Full Outputs)",
    expanded=True,
):
    st.markdown("### 【基准输出 1 & 2】初始与最终可达矩阵")
    df_b_init = ssim_df(base_adj)
    df_b_fin = make_reach_df(base_reach, base_drv, base_dep)
    paired_tables(
        "表 A · 初始可达矩阵（Initial）",
        "只记录问卷中判定的第一层直接关系，尚未做传递闭包推导。",
        df_b_init, "V8.1_基准_初始邻接矩阵.csv", "⬇️ 下载表 A 初始邻接矩阵 CSV",
        "表 B · 最终可达矩阵与驱动力/依赖度（Final）",
        "在表 A 基础上补足传递闭包，并给出驱动力（行和）与依赖度（列和）。",
        df_b_fin, "V8.1_基准_最终可达矩阵.csv", "⬇️ 下载表 B 最终可达矩阵 CSV",
    )

    st.markdown("### 【基准输出 3】层级划分表 (Level Partitioning)")
    df_base_level = pd.DataFrame(level_records(base_levels, "所处层级"))
    with st.container(border=True):
        st.caption("按可达集与前因集迭代剔除得到的层级；层级号越小越靠近结果/表层。")
        st.dataframe(df_base_level, width="stretch")
    spacer(8)
    dl1, _g1, _r1 = st.columns(PAIR_SPEC, gap="small")
    with dl1:
        download_csv(df_base_level, "V8.1_基准_层级划分表.csv", "⬇️ 下载层级划分表 CSV")
    spacer(24)

    st.markdown("### 【基准输出 4 & 5】有向图结构模型与 MICMAC 象限散点图")
    cv1, cv2 = pair_columns()
    with cv1:
        with st.container(border=True):
            st.markdown("**图 A · 问卷基准 M-TISM 有向图**")
            st.caption("节点为 7 大要素，实线为直接关系，虚线为传递闭包间接关系。")
            st.graphviz_chart(
                draw_digraph(base_levels, base_adj, base_reach, "#e2e8f0"),
                width="stretch",
            )
            st.caption(NODE_LEGEND)
    with cv2:
        with st.container(border=True):
            st.markdown("**图 B · 问卷基准 MICMAC 四象限散点图**")
            st.caption("按驱动力/依赖度中位数划分四象限，象限标签固定在四角。")
            fig_b = draw_micmac(base_dep, base_drv, "问卷基准模型")
            st.pyplot(fig_b)
            plt.close(fig_b)
    spacer(24)

    st.markdown("### 【基准输出 6】关系解释表 (Interpretive Matrix)")
    base_interp_rows = [
        {
            "因果路径": f"【E{i + 1}】➔【E{j + 1}】",
            "业务机理说明": f"问卷实证：{FACTORS_7[i]} 对 {FACTORS_7[j]} 具直接驱动与制约作用。",
        }
        for i in range(N7)
        for j in range(N7)
        if base_adj[i][j] == 1 and i != j
    ]
    df_base_interp = pd.DataFrame(base_interp_rows)
    with st.container(border=True):
        st.caption("逐条列出问卷基准中成立的直接因果路径及其业务解释。")
        st.dataframe(df_base_interp, width="stretch")
    spacer(8)
    dl2, _g2, _r2 = st.columns(PAIR_SPEC, gap="small")
    with dl2:
        download_csv(df_base_interp, "V8.1_基准_关系解释矩阵.csv", "⬇️ 下载关系解释矩阵 CSV")
    spacer(10)

st.markdown("---")

# ================= PART 2 =================
st.header("📚 第二部分：专家访谈数据源与 SSIM 判定档案 (Expert Evidence)")
st.markdown(
    "数据源为两份 TISM 专家深度访谈问卷原件，**只读展示、不提供在线录入或配置**。"
    "两位专家的回答已结构化归档为下方的数据源信息概要与 SSIM 判定表。"
)
st.caption(
    "问卷结构：7 个核心变量定义 → 模块一 IF vs IF（6 组）→ 模块二 IF vs CSF（12 组）→ "
    "模块三 CSF vs CSF（3 组）→ 敏捷治理实施路径 5 问 → 因果骨架校验 → 附录 21 组判定汇总矩阵。"
)

tab_e01, tab_e02 = st.tabs([EXPERT_DOSSIER["E01"]["tab"], EXPERT_DOSSIER["E02"]["tab"]])

for tab, key in zip((tab_e01, tab_e02), ("E01", "E02")):
    d = EXPERT_DOSSIER[key]
    with tab:
        st.subheader(f"{d['code']} · {d['name']}")
        st.caption(f"视角：{d['perspective']}")

        st.markdown("**① 数据源信息概要（访谈问卷原件字段）**")
        info_rows = [
            ("专家编号", d["code"]),
            ("角色 / 职务", d["role"]),
            ("核心责任", d["duty"]),
            ("治理边界", d["boundary"]),
            ("监督范围 / 视角限制", d["scope"]),
            ("整理日期", d["date"]),
            ("源文件", d["source"]),
            ("问卷结构", "21 组关系判定（6+12+3）+ 实施路径 5 问 + 因果骨架校验 + 附录汇总矩阵"),
            ("判定统计", d["counts"]),
        ]
        df_info = pd.DataFrame(info_rows, columns=["字段", "原文内容"])
        with st.container(border=True):
            st.dataframe(df_info, width="stretch", hide_index=True)

        spacer(12)
        st.markdown("**② 分析的 SSIM 判定表（7×7，只读）**")
        with st.container(border=True):
            st.caption(SSIM_LEGEND)
            st.dataframe(ssim_df(d["ssim"]), width="stretch", height=290)
        spacer(8)
        de, _ge, _re = st.columns(PAIR_SPEC, gap="small")
        with de:
            download_csv(
                ssim_df(d["ssim"]),
                f"V8.1_{key}_SSIM判定表.csv",
                f"⬇️ 下载 {key} 的 SSIM 判定表 CSV",
            )
        spacer(16)

        with st.expander(f"📋 展开 {key} 的 21 组关系判定明细（与 E02 逐组对照）"):
            df_rel = pd.DataFrame(
                [
                    {
                        "关系代码": code,
                        "因素对": pair,
                        f"{key} 判定": v01 if key == "E01" else v02,
                        "E02 判定（对照）" if key == "E01" else "E01 判定（对照）": v02 if key == "E01" else v01,
                        "一致性": agree,
                    }
                    for code, pair, v01, v02, agree in RELATION_21
                ]
            )
            st.dataframe(df_rel, width="stretch", hide_index=True, height=560)
            download_csv(df_rel, f"V8.1_{key}_21组判定明细.csv", f"⬇️ 下载 {key} 21 组判定明细 CSV")

        with st.expander(f"🧭 展开 {key} 的访谈因果骨架校验要点（原文归纳）"):
            df_sk = pd.DataFrame(d["skeleton"], columns=["校验项", "专家原文归纳"])
            st.dataframe(df_sk, width="stretch", hide_index=True)

st.markdown("---")

# ================= PART 2 · 权重计算依据 =================
st.subheader("📐 专家权重计算依据与完整推演过程")
st.markdown(
    "系数 **0.4 / 0.4 / 0.2** 来自归档代码 `compute_objective_consensus` 的固定设定"
    "（存档记录 V1 第 4.3 节），**是系统设计参数，不是统计估计值**。"
    "下面给出每一档取值、两位专家的实际打分、原始分代入、归一化以及它对成边的实际影响，便于逐项复算。"
)

with st.expander("展开权重计算依据（默认展开）", expanded=True):
    st.markdown("**第 1 步 · 三个维度的系数来源与分档取值标准**")
    df_rules = pd.DataFrame(
        [
            {
                "维度": "从业年限",
                "系数": 0.4,
                "为何占该权重": "经验积累决定对因果关系判断的可靠性，与岗位层级并列为主维度（合计 0.8）",
                "分档取值": "超过 20 年 = 0.5；10–20 年 = 0.4；其余 = 0.2",
            },
            {
                "维度": "岗位层级",
                "系数": 0.4,
                "为何占该权重": "观测位置决定信息覆盖面与判断口径，与从业年限并列为主维度（合计 0.8）",
                "分档取值": "总部高管/区域总监 = 0.5；项目经理/部门主管 = 0.3；其余 = 0.2",
            },
            {
                "维度": "敏捷熟悉度",
                "系数": 0.2,
                "为何占该权重": "对敏捷治理概念的理解仅作辅助校验，故权重减半为 0.2",
                "分档取值": "敏捷实战专家 = 0.5；其余 = 0.3",
            },
        ]
    )
    st.dataframe(df_rules, width="stretch", hide_index=True, height=150)

    spacer(10)
    st.markdown("**第 2 步 · 两位专家按上述标准逐项打分**")
    df_scores = pd.DataFrame(
        [
            {
                "专家": key,
                "从业年限档": EXPERT_DOSSIER[key]["exp"],
                "年限分": EXPERT_DOSSIER[key]["e_score"],
                "岗位层级档": EXPERT_DOSSIER[key]["role_level"],
                "层级分": EXPERT_DOSSIER[key]["r_score"],
                "敏捷档": EXPERT_DOSSIER[key]["agile"],
                "敏捷分": EXPERT_DOSSIER[key]["a_score"],
            }
            for key in ("E01", "E02")
        ]
    )
    st.dataframe(df_scores, width="stretch", hide_index=True, height=110)

    spacer(10)
    st.markdown("**第 3 步 · 原始分代入与归一化（逐项可复算）**")
    st.code(
        "原始分 = 0.4 × 年限分 + 0.4 × 层级分 + 0.2 × 敏捷分\n"
        "\n"
        "E01 = 0.4×0.5 + 0.4×0.3 + 0.2×0.5 = 0.20 + 0.12 + 0.10 = 0.42\n"
        "E02 = 0.4×0.5 + 0.4×0.5 + 0.2×0.5 = 0.20 + 0.20 + 0.10 = 0.50\n"
        "\n"
        "归一化：w_i = 原始分_i / Σ原始分，Σ = 0.42 + 0.50 = 0.92\n"
        "w_E01 = 0.42 / 0.92 = 0.4565\n"
        "w_E02 = 0.50 / 0.92 = 0.5435   （合计 = 1.0000）",
        language="text",
    )

    spacer(10)
    st.markdown("**第 4 步 · 聚合规则与对推演结果的实际影响**")
    st.markdown(
        "- **聚合**：某方向被判 **V 或 X** 则加该专家权重；判 **A** 则加在转置方向；判 **O** 不加分。\n"
        "- **聚合矩阵读法**：`1.000` = 两人一致认定；`0.544` = 仅 E02 认定；`0.456` = 仅 E01 认定；`0` = 均未认定。\n"
        "- **对阈值 0.5 的影响**：E02 独有的判定加权得分 **0.5435 > 0.5**，可独立成边；"
        "E01 独有的判定只有 **0.4565 < 0.5**，无法独立成边。\n"
        "- **权重差异的来源**：两位专家年限、敏捷得分相同，仅岗位层级不同"
        "（E02 为总部高管/区域总监 = 0.5，E01 为项目经理/部门主管 = 0.3），"
        "因此 E02 略高，代表总部监管口径。"
    )
    df_w = pd.DataFrame(
        [
            {
                "专家": key,
                "年限分": EXPERT_DOSSIER[key]["e_score"],
                "层级分": EXPERT_DOSSIER[key]["r_score"],
                "敏捷分": EXPERT_DOSSIER[key]["a_score"],
                "原始分": EXPERT_DOSSIER[key]["raw"],
                "归一化权重": round(EXPERT_DOSSIER[key]["raw"] / 0.92, 4),
            }
            for key in ("E01", "E02")
        ]
    )
    st.dataframe(df_w, width="stretch", hide_index=True, height=110)
    spacer(8)
    dw, _gw, _rw = st.columns(PAIR_SPEC, gap="small")
    with dw:
        download_csv(df_w, "V8.1_专家权重计算依据.csv", "⬇️ 下载权重计算依据表 CSV")

st.markdown("---")

# ================= PART 3 =================
st.header("⚖️ 第三部分：群体问卷与专家缓存数据的综合决策与动态对比 (Synthesis & Decision)")
st.markdown(
    "设置参数后点击执行按钮，系统读取第二部分的两位专家真实 SSIM 判定，"
    "按上方权重聚合后结合第一部分问卷基准输出对比报告。**结果按本次执行时的参数冻结**。"
)

# 控件不使用 st.form：改动会立即触发一次 rerun，从而能真实提示"参数已变更"，
# 并保证「本次生效参数」与当前控件值可区分。若放进 form，改动不提交就永远检测不到。
c_m1, c_m2, c_m3 = st.columns(3)
with c_m1:
    fusion_threshold = st.slider(
        "1. 群体共识截断阈值 (Threshold)", 0.3, 0.9, 0.5, 0.05,
        help="加权得分达到该值的关系才被采纳；调高会减少边、调低会增加边。改动后会立即提示参数已变更，点击执行后刷新报告。",
    )
with c_m2:
    fusion_mode = st.selectbox(
        "2. 多源数据融合策略",
        [
            "【替代模式】完全采纳专家缓存综合意见",
            "【叠加模式】问卷基准 40% + 专家缓存 60% 混合加权",
            "【保守模式】取问卷与专家意见的交集 (AND逻辑)",
        ],
        index=1,
        help="三种模式使用不同的成边公式。切换后会立即提示参数已变更，点击执行后综合可达矩阵与层级随之改变。",
    )
with c_m3:
    output_compare_type = st.multiselect(
        "3. 综合多维输出项多选",
        [
            "1. 综合群体初始与最终可达矩阵",
            "2. 综合群体层级划分表 (Level Partitioning)",
            "3. M-TISM 有向图结构模型 (带变动高亮)",
            "4. MICMAC 象限对比散点图",
            "5. 多专家访谈综合解释矩阵",
        ],
        default=[
            "1. 综合群体初始与最终可达矩阵",
            "2. 综合群体层级划分表 (Level Partitioning)",
            "3. M-TISM 有向图结构模型 (带变动高亮)",
            "4. MICMAC 象限对比散点图",
            "5. 多专家访谈综合解释矩阵",
        ],
        help="勾选哪些项，报告就只显示哪些项；取消勾选后会立即提示参数已变更，点击执行后对应区块消失。",
    )
# 实时预演：控件不在 form 内，任一改动都会 rerun，此行数字会即时变化，
# 让用户无需点执行也能看见阈值 / 融合策略 / 输出项确实在起作用。
_agg_now, _weights_now = compute_objective_consensus(EXPERT_PANEL)
_preview_adj = build_fused_adj(_agg_now, base_adj, fusion_mode, fusion_threshold)
_preview_edges = int(_preview_adj.sum() - N7)
st.caption(
    f"**当前参数实时预演**（随上方控件即时变化）：阈值 `{fusion_threshold}` × "
    f"`{fusion_mode.split('】')[0]}】` → 成边 **{_preview_edges}** 条（不含对角线）；"
    f"已勾选 **{len(output_compare_type)}** 项输出。点击下方执行按钮后，报告按此参数重算并冻结。"
)

submitted_decision = st.button(
    "⚖️ 读取缓存并执行综合决策计算", type="primary", width="stretch",
    help="真正触发一次计算；未点击时下方不产生任何报告。",
)

if "v81_result" not in st.session_state:
    st.session_state["v81_result"] = None

if submitted_decision:
    agg_now, weights_now = compute_objective_consensus(EXPERT_PANEL)
    fused = build_fused_adj(agg_now, base_adj, fusion_mode, fusion_threshold)
    r_reach, r_trans, r_drv, r_dep, r_levels = run_tism_pipeline(fused)
    st.session_state["v81_result"] = {
        "fingerprint": decision_fingerprint(fusion_threshold, fusion_mode, output_compare_type),
        "params": {
            "mode": fusion_mode,
            "threshold": float(fusion_threshold),
            "expert_count": len(EXPERT_PANEL),
            "weights": weights_now,
            "outputs": list(output_compare_type),
        },
        "fused": fused,
        "reach": r_reach,
        "drv": r_drv,
        "dep": r_dep,
        "levels": r_levels,
    }
    st.success("✅ 综合决策计算已完成，本次结果已冻结，详见下方报告。")

st.markdown("---")
result = st.session_state["v81_result"]

if result is None:
    st.subheader("🎯 综合决策模型全套多维输出与动态对比报告")
    st.info(
        "👆 **尚未执行综合决策。** 请在上方设置「共识阈值 / 融合策略 / 输出项」，"
        "然后点击 **「读取缓存并执行综合决策计算」**，本区才会生成报告。"
    )
    st.caption("未执行时此处保持空白，避免把未确认的参数当成已推演结论。")
else:
    current_fp = decision_fingerprint(fusion_threshold, fusion_mode, output_compare_type)
    st.subheader("🎯 综合决策模型全套多维输出与动态对比报告")
    p = result["params"]

    st.markdown("**本次生效参数**（下列值来自上一次点击执行时，不是当前控件值）")
    df_params = pd.DataFrame(
        [
            {"参数": "共识截断阈值", "本次生效值": str(p["threshold"])},
            {"参数": "融合策略", "本次生效值": p["mode"]},
            {"参数": "专家人数", "本次生效值": f"{p['expert_count']} 位（固定 E01 / E02）"},
            {"参数": "归一化权重", "本次生效值": " · ".join(f"{k.split(' ')[0]} = {v:.4f}" for k, v in p["weights"].items())},
            {"参数": "输出项", "本次生效值": "、".join(p["outputs"]) if p["outputs"] else "（未勾选任何输出项）"},
        ]
    )
    st.dataframe(df_params, width="stretch", hide_index=True)

    if current_fp != result["fingerprint"]:
        st.warning(
            "⚠️ 当前「阈值 / 融合策略 / 输出项」已变更，下方仍是**上一次执行**的结果。"
            "请重新点击「读取缓存并执行综合决策计算」以刷新。"
        )

    out = p["outputs"]
    fused = result["fused"]
    r_reach, r_drv, r_dep, r_levels = (
        result["reach"], result["drv"], result["dep"], result["levels"]
    )

    if "1. 综合群体初始与最终可达矩阵" in out:
        st.markdown("#### 📄 综合输出 1 & 2：缓存调用与群体共识可达矩阵")
        df_fused = ssim_df(fused)
        df_r_fin = make_reach_df(r_reach, r_drv, r_dep)
        paired_tables(
            "表 A · 综合群体初始邻接矩阵",
            "按本次阈值与融合策略，逐格过滤后得到的直接关系，尚未做传递闭包。",
            df_fused, "V8.1_综合_初始邻接矩阵.csv", "⬇️ 下载表 A 综合邻接矩阵 CSV",
            "表 B · 综合群体最终可达矩阵与驱动力",
            "在表 A 上补传递闭包，并给出驱动力（行和）与依赖度（列和）。",
            df_r_fin, "V8.1_综合_最终可达矩阵.csv", "⬇️ 下载表 B 综合可达矩阵 CSV",
        )

    if "2. 综合群体层级划分表 (Level Partitioning)" in out:
        st.markdown("#### 📄 综合输出 3：专家群体综合层级划分表")
        df_r_level = pd.DataFrame(level_records(r_levels, "综合所处层级"))
        with st.container(border=True):
            st.caption("与基准层级表对照，可观察专家意见引入后层级结构的变化。")
            st.dataframe(df_r_level, width="stretch")
        spacer(8)
        dl3, _g3, _r3 = st.columns(PAIR_SPEC, gap="small")
        with dl3:
            download_csv(df_r_level, "V8.1_综合_层级划分表.csv", "⬇️ 下载综合层级划分表 CSV")
        spacer(24)

    if (
        "3. M-TISM 有向图结构模型 (带变动高亮)" in out
        or "4. MICMAC 象限对比散点图" in out
    ):
        st.markdown("#### 🖼️📈 综合输出 4 & 5：M-TISM 有向图结构模型与 MICMAC 象限对比")
        st.info(
            "🎨 **色彩语义**：黑色实线 = 问卷基准与专家共识一致的核心机理；"
            "红色粗线 = 引入专家访谈后新增的隐性因果传导路径；灰色虚线 = 传递闭包推导出的间接路径。"
        )
        c_out3, c_out4 = pair_columns()
        with c_out3:
            with st.container(border=True):
                if "3. M-TISM 有向图结构模型 (带变动高亮)" in out:
                    st.markdown("**图 A · 综合 M-TISM 有向图（多源比对视图）**")
                    st.caption("红线为专家意见带来的新增路径，便于与基准图对照。")
                    st.graphviz_chart(
                        draw_digraph(r_levels, fused, r_reach, "#fef3c7", base_adj=base_adj),
                        width="stretch",
                    )
                    st.caption(NODE_LEGEND)
        with c_out4:
            with st.container(border=True):
                if "4. MICMAC 象限对比散点图" in out:
                    st.markdown("**图 B · MICMAC 驱动力-依赖度四象限对比**")
                    st.caption("蓝三角为问卷基准、橙圆点为专家综合模型，图例位于图下方。")
                    fig2 = draw_micmac(r_dep, r_drv, "专家综合模型", base_dep, base_drv)
                    st.pyplot(fig2)
                    plt.close(fig2)
        spacer(24)

    if "5. 多专家访谈综合解释矩阵" in out:
        st.markdown("#### 📖 综合输出 6：多专家访谈语料库与决策解释矩阵")
        expert_matrix_rows = [
            {
                "因果路径": f"【E{i + 1}】➔【E{j + 1}】",
                "涉及要素": f"{FACTORS_7[i]} ➔ {FACTORS_7[j]}",
                "群体决策结论": (
                    f"{p['expert_count']} 位专家加权融合（阈值 {p['threshold']}）认定该路径成立；"
                    "建议纳入敏捷监管阻断清单。"
                ),
            }
            for i in range(N7)
            for j in range(N7)
            if fused[i][j] == 1 and i != j
        ]
        df_exp_matrix = pd.DataFrame(expert_matrix_rows)
        with st.container(border=True):
            st.caption("逐条列出本次综合模型中成立的因果路径与解释，可整表下载。")
            st.dataframe(df_exp_matrix, width="stretch", height=350)
        spacer(8)
        dl4, _g4, _r4 = st.columns(PAIR_SPEC, gap="small")
        with dl4:
            download_csv(df_exp_matrix, "V8.1_综合_决策解释矩阵.csv", "⬇️ 下载决策解释矩阵 CSV")
        spacer(16)

    if st.button("🗑️ 清除本次计算结果", width="stretch", help="清空冻结结果，报告区回到未执行态。"):
        st.session_state["v81_result"] = None
        st.rerun()

st.markdown("---")
st.caption(
    "V8.1 · 双表分离对齐 + 真实专家数据档案 + 按钮全可用 ｜ 刘总论文实证研究支撑系统 ｜ "
    "Multi-Expert M-TISM Decision Framework"
)
