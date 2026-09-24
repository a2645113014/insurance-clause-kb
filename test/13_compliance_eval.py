"""合规判定评测 —— Phase 3.2 的效果验证入口

为什么必须分三档出题
--------------------
合规审查的错误**不是对称的**：
- 把违规判成「合规」→ 放过一个风险条款（漏检）
- 把合规判成「违规」→ 冤枉一个正常条款（误报），业务上同样要命
一刀切的准确率会掩盖方向。所以金标集分三档，指标分开算：

| 档位 | 含义 | 关键指标 |
|---|---|---|
| violation | 应判违规（金标条号明确） | 违规召回率、条号引用命中率 |
| compliant | 应判合规（正常条款表述） | **误判违规率**（最危险的一侧）、判合规率 |
| boundary | 主观边界，不计分只观察 | 模型是否敢下判定 |

为什么要把「检索召回」与「模型引用」拆成两个指标
----------------------------------------------
一条正例没被判违规，可能死在两个完全不同的地方：
1. **金标条号压根没被召回** → 检索层问题（top_k 太小 / 兜底没开）
2. **召回了，模型没认出来或引用了别条** → 判定层问题（prompt / 模型能力）
合成一个「准确率」就分不清该改哪层。所以：
- `recall_hit`：金标条号是否出现在召回列表里 ← 检索层责任
- `cited_gold`：模型给出的 violated_item_no 是否等于金标 ← 判定层责任

只算召回指标的消融不需要调 LLM（省 90% 时间），走 `--recall-only`。

第四个指标是**编造检测**：`violated_item_no` 不在召回列表里 = 模型无中生有。
prompt 硬性规则第 1 条明令禁止，但禁止归禁止，得有个哨兵盯着。

⚠️ 二值指标会饱和，必须配一个连续量
------------------------------------
`retrieval_recall_rate` 只统计计分的 14 条正例 —— 实测消融 6 组全是 1.0，
六组配置在这个指标上完全不可区分，「top_k 该取几」这种问题**根本答不出来**。
所以另加两个口径：
- `retrieval_recall_rate_all`：分母含边界组的 3 条金标，关掉某条兜底路才会露出来
- `gold_avg_rank`：金标在清单里的位次（连续量），能看出兜底条目把金标冲到了第几位
两个新口径都是**检索层**指标，与判定分组无关，所以把边界组纳进来是合理的
—— 边界组「不计判定分」不代表它的金标没有检索意义。

用法（在项目根目录执行）
----------------------
    .venv\\Scripts\\python.exe test/13_compliance_eval.py                      # base 配置跑全流程
    .venv\\Scripts\\python.exe test/13_compliance_eval.py --recall-only        # 只算召回（快）
    .venv\\Scripts\\python.exe test/13_compliance_eval.py --ablation           # 跑一组配置做对照
    .venv\\Scripts\\python.exe test/13_compliance_eval.py --write-cases        # 把内置金标集落盘供人工改

结果落盘 `data/eval/compliance_eval_<signature><模式后缀>.json`：
全流程不带后缀（基线，最贵、最难被覆盖），`--recall-only` 带 `_recall`，
`--ablation` 带 `_ablation`。三者结论不同，共用文件名会互相覆盖。
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Windows 控制台默认 GBK，print 里的符号会抛 UnicodeEncodeError 打断脚本。
# ⚠️ 只设 errors，**不设 encoding** —— 改成 utf-8 会让 PowerShell 管道的中文
# 全变乱码（管道按控制台代码页解码）。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 沙箱代理会劫持到本地服务与本机模型的连接，必须在导入客户端前清掉
for _k in list(os.environ):
    if _k.lower() in {"http_proxy", "https_proxy", "all_proxy", "grpc_proxy", "no_proxy"}:
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from app.conf.compliance_config import compliance_config
from app.core.load_prompt import load_prompt
from app.tool.compliance_tools import build_negative_list_block, check_compliance

EVAL_DIR = os.path.join(ROOT, "data", "eval")
CASES_FILE = os.path.join(EVAL_DIR, "compliance_cases.jsonl")
CACHE_FILE = os.path.join(EVAL_DIR, "compliance_eval_cache.jsonl")

# 只保留第 1/2 大类的配置名 —— 见 CASES 上方说明
CATEGORIES_TEXT_ONLY = ["1", "2"]

# ============================================================ 金标集
# 正例的判定依据写进 note，便于人工复核时对着负面清单原文核。
# `alt_item_nos` 是「同样说得通」的备选条号：一条表述触碰多条是常态，
# 只认单一条号会把正确答案判成错误。只要引用落在 {gold} ∪ alt 里就算命中。
#
# ⚠️ 一个必须写下来的语料事实：负面清单第 3 大类（费率厘定及精算假设，36 条）
# 与第 4 大类（产品报送管理，19 条）合计 55 条，判的是**精算报告与备案材料**，
# 条款文本无从对应。所以本评测集的正例只取自第 1、2 大类。把 55 条费率/报送
# 条目混进候选池，对条款审查只有增加误判风险这一个作用 —— 这是 `cat12`
# 那个消融配置要量的事。
CASES = [
    # ---------------------------------------------- violation（应判违规）
    {
        "id": "V01", "group": "violation", "product_scope": None, "gold_item_no": 12,
        "clause_text": "申请身故保险金时，申请人除提供死亡证明或户籍注销证明外，"
                       "还须提供火化证明与丧葬证明。",
        "note": "条12 明确列举「人寿保险条款约定身故保险金申请除死亡证明/户籍注销证明外，"
                "还需提供火化证明、丧葬证明」为不合理材料要求。",
    },
    {
        "id": "V02", "group": "violation", "product_scope": None, "gold_item_no": 15,
        "clause_text": "被保险人身故后，本公司不一次性全额给付身故保险金，"
                       "而是按本合同约定的标准分期给付生存金给受益人。",
        "note": "条15 列举「身故后不全额给付，按约定标准分期给付生存金给受益人」。",
    },
    {
        "id": "V03", "group": "violation", "product_scope": None, "gold_item_no": 15,
        "clause_text": "被保险人经医院确诊罹患本合同所载疾病的，"
                       "需自确诊之日起生存满三十日，本公司方给付保险金。",
        "note": "条15 列举「疾病保险约定确诊后需生存一定期限方可获得保险金给付」。",
    },
    {
        "id": "V04", "group": "violation", "product_scope": "医疗保险", "gold_item_no": 16,
        "clause_text": "本产品保险期间届满时，本公司如未收到投保人的不续保申请，"
                       "则视同续保，投保人应按约定交纳续期保险费。",
        "note": "条16「届满时未收到不续保申请则视同续保，侵害消费者选择权」。",
    },
    {
        "id": "V05", "group": "violation", "product_scope": "护理保险", "gold_item_no": 21,
        "clause_text": "因细菌或病毒感染引发的保险事故，本公司不承担给付保险金的责任。",
        "note": "条21 列举护理保险约定对细菌或病毒感染免责。",
    },
    {
        "id": "V06", "group": "violation", "product_scope": None, "gold_item_no": 13,
        "clause_text": "除另有约定外，本合同身故保险金的第一受益人为贷款发放机构。",
        "note": "条13 列举受益人表述为「第一受益人为贷款发放机构」。",
    },
    {
        "id": "V07", "group": "violation", "product_scope": None, "gold_item_no": 8,
        "clause_text": "本合同项下保险金请求权的诉讼时效为一年，"
                       "自被保险人或受益人知道保险事故发生之日起计算。",
        "note": "条8 约定诉讼时效与《保险法》不一致（法定为三年）。",
    },
    {
        "id": "V08", "group": "violation", "product_scope": None, "gold_item_no": 10,
        "clause_text": "因本合同发生的争议，双方约定由保险人住所地人民法院"
                       "或保险单签发地人民法院管辖。",
        "note": "条10 约定管辖法院范围与《民事诉讼法》地域管辖规定不符。",
    },
    {
        "id": "V09", "group": "violation", "product_scope": "短期健康保险", "gold_item_no": 18,
        "clause_text": "本产品为短期健康保险产品，"
                       "续保时本公司有权根据医疗费用水平变化调整保险费率。",
        "note": "条18 短期健康保险条款含续保时可能调整费率的表述。",
    },
    {
        "id": "V10", "group": "violation", "product_scope": "健康保险", "gold_item_no": 19,
        "clause_text": "投保人申请解除本合同时，应当同时申请解除本合同的附加险合同，"
                       "不得单独解除附加险。",
        "note": "条19 约定消费者不得单独解除附加险。",
    },
    {
        "id": "V11", "group": "violation", "product_scope": None, "gold_item_no": 41,
        "alt_item_nos": [35, 59],
        "clause_text": "被保险人在等待期内发生保险事故的，"
                       "本公司按已交保险费的百分之五十退还，不承担给付保险金责任。",
        "note": "条41 通过等待期内不全额退还保费变相惩罚消费者。",
    },
    {
        "id": "V12", "group": "violation", "product_scope": "医疗保险", "gold_item_no": 27,
        "clause_text": "本合同项下处方审核由本公司委托的第三方医疗服务机构负责，"
                       "审核结果以该机构意见为准。",
        "note": "条27（2026 版新增）处方审核主体约定为第三方服务商，"
                "且未明确列明保险公司应承担的审核责任。",
    },
    {
        "id": "V13", "group": "violation", "product_scope": "医疗保险", "gold_item_no": 45,
        "alt_item_nos": [21],
        "clause_text": "被保险人用药时长符合慈善赠药申请条件，但因未提交相关申请"
                       "或申请材料不全导致慈善赠药申请未通过的，"
                       "该部分药品费用本公司不承担给付责任。",
        "note": "条45 责任免除包含慈善赠药未通过的费用，涉嫌加重被保险人义务。",
    },
    {
        "id": "V14", "group": "violation", "product_scope": None, "gold_item_no": 12,
        "clause_text": "意外身故保险金申请除提供公安交通管理部门出具的事故认定书外，"
                       "还须提供当次交通工具客票存根。",
        "note": "条12 列举意外伤害保险要求提供客票（存根）等不合理材料。",
    },
    # ---------------------------------------------- compliant（正常表述）
    {
        "id": "C01", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "自投保人签收保险单之日起十五日内为犹豫期。"
                       "投保人在犹豫期内要求解除本合同的，本公司无息退还已交保险费。",
        "note": "犹豫期与全额退费的规范写法，无对应禁止性条目。",
    },
    {
        "id": "C02", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "本合同的保险期间为十一年，自本合同生效之日起计算。",
        "note": "保险期间约定的规范写法。",
    },
    {
        "id": "C03", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "投保人于犹豫期后要求解除本合同的，"
                       "本公司自收到解除合同通知书之日起三十日内，"
                       "向投保人退还本合同当时的现金价值。",
        "note": "退保与现金价值条款的规范写法（与《保险法》一致）。",
    },
    {
        "id": "C04", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "投保人或被保险人可以指定一人或数人为身故保险金受益人。"
                       "受益人为数人的，可以确定受益顺序和受益份额。",
        "note": "受益人指定的规范写法，未出现「贷款发放机构」等违规表述。",
    },
    {
        "id": "C05", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "因下列情形之一导致被保险人身故的，本公司不承担给付保险金的责任："
                       "一、投保人对被保险人的故意杀害、故意伤害；"
                       "二、被保险人故意犯罪或抗拒依法采取的刑事强制措施；"
                       "三、被保险人自本合同成立之日起二年内自杀。",
        "note": "免责条款集中列明的规范写法（正合条2「应统一、集中」的要求）。",
    },
    {
        "id": "C06", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "申请身故保险金时，申请人须填写保险金给付申请书，"
                       "并提供下列证明和资料：一、保险合同；二、申请人的有效身份证件；"
                       "三、国家卫生行政部门规定的医疗机构出具的死亡证明。",
        "note": "理赔材料的规范清单，不含火化证明/丧葬证明等不合理要求。",
    },
    {
        "id": "C07", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "本公司收到保险金给付申请书及本合同约定的证明和资料后，"
                       "将在五个工作日内作出核定；情形复杂的，在三十日内作出核定。",
        "note": "理赔时限的规范写法（符合《保险法》第二十三条）。",
    },
    {
        "id": "C08", "group": "compliant", "product_scope": None, "gold_item_no": None,
        "clause_text": "本合同的基本保险金额为人民币十万元，"
                       "由投保人与本公司在投保时约定并在保险单上载明。",
        "note": "保险金额约定的规范写法（对应条7 要求的与《保险法》概念一致）。",
    },
    {
        "id": "C09", "group": "compliant", "product_scope": "短期健康保险", "gold_item_no": None,
        "clause_text": "本产品保险期间为一年。保险期间届满，"
                       "投保人需要重新向本公司申请投保本产品，"
                       "经本公司同意并交纳保险费后，方可获得新的保险合同。",
        "note": "不保证续保的正确表述方式（条22 批评的是错误表述，不是不保证续保本身）。",
    },
    {
        "id": "C10", "group": "compliant", "product_scope": "健康保险", "gold_item_no": None,
        "clause_text": "既往症指在本合同生效日之前，被保险人已患且已知晓的疾病。",
        "note": "条24 要求的既往症标准定义表述，逐字合规。",
    },
    # ---------------------------------------------- boundary（只观察不计分）
    {
        "id": "B01", "group": "boundary", "product_scope": "医疗保险", "gold_item_no": 33,
        "clause_text": "本合同的年度免赔额为一万元，"
                       "被保险人自社会医疗保险获得的补偿金额可用于抵扣免赔额。",
        "note": "条33「医疗保险设置过高的免赔额」—— 一万元是否算过高是主观判断，"
                "观察模型是敢判还是判无法判定。",
    },
    {
        "id": "B02", "group": "boundary", "product_scope": "健康保险", "gold_item_no": 35,
        "clause_text": "本健康保险产品的等待期为一百八十日。",
        "note": "条35「等待期设置过长」—— 时长缺失对照基准，属主观边界。",
    },
    {
        "id": "B03", "group": "boundary", "product_scope": "年金保险", "gold_item_no": 32,
        "alt_item_nos": [47, 49],
        "clause_text": "本合同提供加保与减保功能，"
                       "投保人可在保险期间内申请增加或减少本合同的基本保险金额。",
        "note": "条32/47「加减保功能实现类万能型自由领取」—— 是否异化取决于"
                "费率与比例约定，单看这句无法断定。",
    },
]


# ============================================================ 工具
def norm_text(text):
    """归一化空白与全角标点，供做 cache key 与比对。"""
    return re.sub(r"\s+", "", text or "")


def case_key(config_name, case):
    raw = f"{config_name}|{case['id']}|{norm_text(case['clause_text'])}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache():
    """读结果缓存。键为 case_key，值为 LLM 原始输出。

    为什么要缓存：一条判定的成本 = 1 次 embedding + 1 次 qwen-max 调用（数秒）。
    跑消融时同一配置要反复调，没缓存会让人放弃做对照。
    """
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    cache[row["key"]] = row
                except Exception:
                    continue
    return cache


def append_cache(rows):
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_verdict(raw):
    """从模型输出里抠出 JSON。容忍代码块包裹与前后废话。

    解析失败**不当作「合规」** —— 那是把格式错误洗成合规结论。
    返回 None，由上层记成解析失败并列进「无法判定」以外的独立桶。
    """
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 退一步：找第一个大括号到最后一个大括号
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def call_llm(prompt, llm):
    """调一次 LLM 拿原始文本输出。"""
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", None)
    if isinstance(content, list):
        # 部分网关返回分段内容列表，拼回字符串
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content or ""


# ============================================================ 单配置跑一轮
def run_config(cases, config_name, overrides, recall_only, use_cache):
    """跑一个配置，返回 (rows, summary)。"""
    llm = None
    if not recall_only:
        from app.lm.lm_utils import get_llm_client

        llm = get_llm_client(json_mode=True)

    cache = load_cache() if use_cache else {}
    new_cache_rows = []
    rows = []

    for case in cases:
        started = time.time()
        key = case_key(config_name, case)
        res = check_compliance(
            case["clause_text"], case.get("product_scope"), **overrides
        )
        retrieved_nos = [it["item_no"] for it in res["items"]]
        gold = case.get("gold_item_no")
        # 金标 + 备选都算命中。括号不能省：`a | b if c else d` 虽然能正确
        # 解析成 `(a | b) if c else d`，但读的人得先想一遍优先级。
        accepted = ({gold} | set(case.get("alt_item_nos") or [])) if gold else set()

        row = {
            "id": case["id"],
            "group": case["group"],
            "gold_item_no": gold,
            "retrieved": len(retrieved_nos),
            "recall_hit": (gold in retrieved_nos) if gold else None,
            # 金标在清单里的位次（1 起）。召回率是二值的，一旦多组配置都命中
            # 就完全分不出好坏 —— 实测消融 6 组召回率全是 1.0。位次是连续量，
            # 才能看出「兜底把清单填到 40 条」这类问题把金标冲到了第几位。
            "gold_rank": (
                retrieved_nos.index(gold) + 1
                if gold and gold in retrieved_nos else None
            ),
            "retrieved_nos": retrieved_nos,
            "verdict": None,
            "cited_item_no": None,
            "cited_in_retrieved": None,
            "cited_gold": None,
            "rag_error": res.get("note") or "",
        }

        if not recall_only:
            cached = cache.get(key)
            if cached is not None:
                raw = cached.get("raw", "")
            else:
                block = build_negative_list_block(res)
                prompt = load_prompt(
                    "compliance_check",
                    clause_text=case["clause_text"],
                    negative_list_items=block,
                )
                try:
                    raw = call_llm(prompt, llm)
                except Exception as e:
                    raw = ""
                    row["rag_error"] = f"{row['rag_error']} | LLM 调用失败：{e}".strip(" |")
                new_cache_rows.append({
                    "key": key,
                    "config": config_name,
                    "case_id": case["id"],
                    "raw": raw,
                })

            parsed = parse_verdict(raw)
            if parsed is None:
                row["verdict"] = "解析失败"
                row["raw_tail"] = (raw or "")[-160:]
            else:
                row["verdict"] = parsed.get("verdict")
                row["cited_item_no"] = parsed.get("violated_item_no")
                row["reason"] = parsed.get("reason")
                row["category"] = parsed.get("category")

            cited = row["cited_item_no"]
            if cited is not None:
                try:
                    cited = int(cited)
                except Exception:
                    cited = None
            row["cited_item_no"] = cited
            row["cited_in_retrieved"] = (cited in retrieved_nos) if cited is not None else None
            row["cited_gold"] = (cited in accepted) if (cited is not None and accepted) else None

        row["elapsed_sec"] = round(time.time() - started, 1)
        rows.append(row)
        flag = ""
        if row["group"] == "violation":
            flag = "V" if row["verdict"] == "违规" else "x"
        elif row["group"] == "compliant":
            flag = "!" if row["verdict"] == "违规" else "."
        print(f"  [{config_name}] {row['id']:>3} {row['group']:<9} "
              f"召回{row['retrieved']:>2}条 "
              f"recall_hit={row['recall_hit']} verdict={row['verdict']} "
              f"cite={row['cited_item_no']} {flag} ({row['elapsed_sec']}s)")

    if new_cache_rows:
        append_cache(new_cache_rows)

    return rows, summarize(rows)


def summarize(rows):
    """按档位汇总指标。分母为 0 的指标填 None，不填 0 —— 0 会被误读成「全错」。"""
    viol = [r for r in rows if r["group"] == "violation"]
    comp = [r for r in rows if r["group"] == "compliant"]
    bnd = [r for r in rows if r["group"] == "boundary"]

    def rate(num, den):
        return round(num / den, 4) if den else None

    v_viol = sum(1 for r in viol if r["verdict"] == "违规")
    v_unknown = sum(1 for r in viol if r["verdict"] == "无法判定")
    v_compliant = sum(1 for r in viol if r["verdict"] == "合规")
    recall_hit = sum(1 for r in viol if r["recall_hit"])
    # 同上：只算正例会 14/14 满格，六组配置全一样。全金标口径（含边界）才有区分度
    recall_hit_all = sum(1 for r in rows if r["recall_hit"])
    cited_gold = sum(1 for r in viol if r["cited_gold"])
    # 位次刻意统计**所有带金标的用例**（含边界组），不只用计分的正例。
    # 原因：14 条正例的金标全在第 1~2 位，六组配置算出来都是 1.50 —— 又饱和了，
    # 旋钮照样调不动。边界组虽然不计判定分，但它带金标（如 B01 的条 33 在
    # 关掉通用兜底后会掉出前 12），是唯一能把位次撑开的样本。检索层质量与
    # 判定分组无关，所以这个口径是合理的。
    ranks = [r["gold_rank"] for r in rows if r.get("gold_rank")]
    # 分母必须用「带金标的用例数」，不能用「命中的用例数」——
    # 用后者的话漏召回的那条会被排除出分母，召回率永远是 100%，指标失效。
    n_gold = sum(1 for r in rows if r.get("gold_item_no"))

    c_fp = sum(1 for r in comp if r["verdict"] == "违规")
    c_ok = sum(1 for r in comp if r["verdict"] == "合规")
    c_unknown = sum(1 for r in comp if r["verdict"] == "无法判定")

    fabricated = [r["id"] for r in rows if r["cited_in_retrieved"] is False]
    parse_fail = [r["id"] for r in rows if r["verdict"] == "解析失败"]

    return {
        "n_violation": len(viol),
        "violation_rate": rate(v_viol, len(viol)),
        "violation_unknown_rate": rate(v_unknown, len(viol)),
        "violation_laundered_rate": rate(v_compliant, len(viol)),
        "retrieval_recall_rate": rate(recall_hit, len(viol)),
        "retrieval_recall_rate_all": rate(recall_hit_all, n_gold),
        "gold_avg_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "n_gold_ranked": len(ranks),
        "avg_retrieved": round(sum(r["retrieved"] for r in rows) / len(rows), 1) if rows else None,
        "citation_hit_rate": rate(cited_gold, len(viol)),
        "n_compliant": len(comp),
        "false_positive_rate": rate(c_fp, len(comp)),
        "compliant_ok_rate": rate(c_ok, len(comp)),
        "compliant_unknown_rate": rate(c_unknown, len(comp)),
        "n_boundary": len(bnd),
        "fabricated_citations": fabricated,
        "parse_failures": parse_fail,
    }


def print_summary(name, summary):
    print("-" * 78)
    print(f"### 配置 {name}")
    print(f"  正例 {summary['n_violation']} 条 | 负例 {summary['n_compliant']} 条 "
          f"| 边界 {summary['n_boundary']} 条")
    print(f"  [检索层] 金标条号召回率      : {summary['retrieval_recall_rate']}")
    print(f"  [检索层] 金标平均位次        : {summary.get('gold_avg_rank')}"
          f"（{summary.get('n_gold_ranked')} 条带金标用例）  <-- 召回率饱和时看它")
    print(f"  [成本]   平均召回条数        : {summary.get('avg_retrieved')}")
    print(f"  [判定层] 违规判定率          : {summary['violation_rate']}")
    print(f"  [判定层] 条号引用命中率      : {summary['citation_hit_rate']}")
    print(f"  [判定层] 正例被判\"合规\"（洗白）: {summary['violation_laundered_rate']}")
    print(f"  [判定层] 正例判\"无法判定\"     : {summary['violation_unknown_rate']}")
    print(f"  [安全]   负例误判违规率      : {summary['false_positive_rate']}  <-- 越低越好")
    print(f"  [安全]   负例判合规率        : {summary['compliant_ok_rate']}")
    print(f"  [安全]   负例判无法判定率    : {summary['compliant_unknown_rate']}")
    if summary["fabricated_citations"]:
        print(f"  [!!] 编造条号（引用了未召回的条号）: {summary['fabricated_citations']}")
    if summary["parse_failures"]:
        print(f"  [!!] 输出解析失败: {summary['parse_failures']}")


def main():
    ap = argparse.ArgumentParser(description="合规判定评测（Phase 3.2）")
    ap.add_argument("--recall-only", action="store_true",
                    help="只算召回指标，不调 LLM（快）")
    ap.add_argument("--ablation", action="store_true",
                    help="跑一组配置做对照（默认 only-recall）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--no-cache", action="store_true", help="忽略结果缓存，强制重跑")
    ap.add_argument("--write-cases", action="store_true", help="把内置金标集落盘后退出")
    args = ap.parse_args()

    if args.write_cases:
        os.makedirs(EVAL_DIR, exist_ok=True)
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            for case in CASES:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"金标集已落盘：{CASES_FILE}（{len(CASES)} 条）")
        return 0

    cases = CASES[: args.limit] if args.limit else CASES
    print("=" * 78)
    print(f"合规判定评测 | 配置指纹={compliance_config.signature()}")
    print(f"  用例数={len(cases)} | 模式={'recall-only' if args.recall_only else '全流程'}"
          f" | 缓存={'off' if args.no_cache else 'on'}")

    # 消融配置：每个都只动一个变量，否则出问题分不清是哪个造成的
    if args.ablation:
        plans = [
            ("base", {}, True),
            ("k6", {"top_k": 6}, True),
            ("k24", {"top_k": 24}, True),
            ("no_universal", {"include_universal": False}, True),
            ("no_scope", {"include_scope_match": False}, True),
            ("cat12", {"categories": CATEGORIES_TEXT_ONLY}, True),
        ]
    else:
        plans = [("base", {}, args.recall_only)]

    all_summaries = {}
    for name, overrides, recall_only in plans:
        print("=" * 78)
        print(f">>> 配置 {name} | overrides={json.dumps(overrides, ensure_ascii=False)}")
        rows, summary = run_config(
            cases, name, overrides, recall_only=recall_only, use_cache=not args.no_cache
        )
        print_summary(name, summary)
        all_summaries[name] = {"overrides": overrides, "summary": summary, "rows": rows}

    # 对照表
    if len(all_summaries) > 1:
        print("=" * 78)
        print("### 召回层对照（只看检索，与判定质量无关）")
        header = (f"{'配置':<14}{'召回率正例':>11}{'召回率全金标':>13}"
                  f"{'平均条数':>10}{'金标位次':>10}")
        print("  " + header)
        for name, blob in all_summaries.items():
            s = blob["summary"]
            avg = round(sum(r["retrieved"] for r in blob["rows"]) / max(1, len(blob["rows"])), 1)
            rank = s.get("gold_avg_rank")
            print(f"  {name:<14}{str(s['retrieval_recall_rate']):>11}"
                  f"{str(s.get('retrieval_recall_rate_all')):>13}{avg:>10}"
                  f"{('-' if rank is None else f'{rank:.2f}'):>10}")
        print("  注：「召回率正例」只统计计分的 14 条，六组实测全 1.0 —— 饱和，没有区分度；")
        print("      「召回率全金标」含边界组 3 条，才能看出关掉某个兜底路会漏哪条；")
        print("      「金标位次」是连续量，位次越低说明金标越靠前、兜底噪音干扰越小。")

    report = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_signature": compliance_config.signature(),
        "recall_only": args.recall_only,
        "configs": {
            name: {"overrides": blob["overrides"], "summary": blob["summary"]}
            for name, blob in all_summaries.items()
        },
        "rows": {name: blob["rows"] for name, blob in all_summaries.items()},
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    # 文件名必须带上运行模式：全流程（调 LLM，贵）与消融 / 只看召回（便宜）
    # 的结论完全不同，共用同一个文件名会让后跑的静默覆盖先跑的
    # —— 实测 `--ablation` 把 base 全流程的报告冲掉过一次，调 LLM 花掉的
    # 那几分钟就白费了。基线报告（不带后缀）是最贵的那份，必须最难被覆盖。
    if args.ablation:
        mode_tag = "_ablation"
    elif args.recall_only:
        mode_tag = "_recall"
    else:
        mode_tag = ""
    out = os.path.join(
        EVAL_DIR, f"compliance_eval_{compliance_config.signature()}{mode_tag}.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("=" * 78)
    print(f"报告已落盘：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
