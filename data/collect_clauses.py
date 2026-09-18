# -*- coding: utf-8 -*-
"""
保险条款语料采集器（第 1 轮）
--------------------------------------------------
职责：从保司「公开信息披露」页抓取条款 PDF → 落盘 data/clauses/pdf/ → 写 sources.csv
设计：幂等（同名文件已存在且大小一致则跳过）、限速 1s/次、只采公开披露页
用法：.venv\\Scripts\\python.exe data\\collect_clauses.py
"""
import csv
import hashlib
import io
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote

import requests
import urllib3

urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "clauses" / "pdf"
SRC_CSV = ROOT / "data" / "clauses" / "sources.csv"
PDF_DIR.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
BASE_H = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}

LOG = []


def log(s=""):
    LOG.append(str(s))


# ----------------------------------------------------------------------
# 险种归类（写进 sources.csv，供后续按险种分层抽样）
# ----------------------------------------------------------------------
KUANGZHONG = [
    ("重大疾病保险", "重疾"), ("疾病保险", "重疾"),
    ("医疗费用保险", "医疗"), ("医疗保险", "医疗"), ("医疗险", "医疗"),
    ("护理保险", "护理"),
    ("年金保险", "年金"), ("养老年金", "年金"),
    ("两全保险", "两全"),
    ("定期寿险", "定寿"),
    ("终身寿险", "终身寿"), ("终身寿", "终身寿"),
    ("意外伤害保险", "意外"), ("意外险", "意外"),
]


def guess_kind(name: str) -> str:
    for kw, label in KUANGZHONG:
        if kw in name:
            return label
    return "其他"


# ----------------------------------------------------------------------
# 0. MinerU 云端可用性体检（导入链路唯一外部依赖，先验它）
# ----------------------------------------------------------------------
def check_mineru():
    log("=" * 72)
    log("### 0. MinerU 云端 API 体检")
    try:
        sys.path.insert(0, str(ROOT))
        from app.conf.mineru_config import mineru_config
        token, base = mineru_config.api_key, mineru_config.base_url
        log(f"base_url = {base}")
        log(f"token    = {token[:6]}...  (len={len(token)})")
        r = requests.post(
            f"{base}/file-urls/batch",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"},
            json={"files": [{"name": "probe.pdf", "data_id": "probe"}],
                  "model_version": "vlm"},
            timeout=30,
        )
        log(f"http status = {r.status_code}")
        try:
            j = r.json()
        except Exception:
            log(f"resp head = {r.text[:300]}")
            return
        log(f"code = {j.get('code')}   msg = {j.get('msg')}")
        if j.get("code") == 0:
            log(">>> 判定：TOKEN 有效，上传通道正常（未真正上传文件，不消耗额度）")
        else:
            log(">>> 判定：**TOKEN 或额度异常**，需去 mineru.net 复查")
    except Exception as e:
        log(f"EXCEPTION {type(e).__name__}: {e}")
        log(">>> 判定：**体检失败**")


# ----------------------------------------------------------------------
# 1. 太平洋人寿：HTML 表格，四列元数据齐全（元数据模板首选）
# ----------------------------------------------------------------------
CPIC_PAGE = "https://www.cpic.com.cn/xrsbx/gkxxpl/hlwbxxx/hlwbxcpxx/zbbxcpxx/"


def fetch_cpic():
    log("")
    log("=" * 72)
    log("### 1. 太平洋人寿 互联网保险产品信息")
    r = requests.get(CPIC_PAGE, headers=BASE_H, timeout=30, verify=False)
    r.encoding = r.apparent_encoding or "utf-8"
    log(f"page status={r.status_code} len={len(r.content)}")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")
    items = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) != 4:
            continue
        name = tds[0].get_text(strip=True)
        a = tds[0].find("a") or tds[1].find("a")
        href = a.get("href") if a else None
        if not href or not name or name == "备案产品名称":
            continue
        items.append({
            "insurer": "太平洋人寿",
            "name": name,
            "url": urljoin(CPIC_PAGE, href),
            "doc_no": tds[2].get_text(strip=True),
            "reg_no": tds[3].get_text(strip=True),
            "page": CPIC_PAGE,
            "kind": guess_kind(name),
        })
    log(f"解析到 {len(items)} 个产品条目")
    for it in items:
        log(f"  [{it['kind']}] {it['name']}  <- {it['url'].rsplit('/',1)[-1]}")
    return items


# ----------------------------------------------------------------------
# 2. 中英人寿：列表页，中文文件名直链（只取「条款」，跳过「产品说明书」）
# ----------------------------------------------------------------------
AC_PAGE = ("https://www.aviva-cofco.com.cn/website/xxzx/gkxxpl/zxxx/"
           "gryljzq/xxpl1/list-1.shtml")


def fetch_aviva():
    log("")
    log("=" * 72)
    log("### 2. 中英人寿 产品条款披露")
    r = requests.get(AC_PAGE, headers=BASE_H, timeout=30, verify=False)
    r.encoding = r.apparent_encoding or "utf-8"
    log(f"page status={r.status_code} len={len(r.content)}")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")
    seen, items = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        if "条款" not in href:
            continue
        # 去掉历史版本（带日期区间的旧版），只留最新一份
        if re.search(r"_\d{8}-\d{8}\.pdf$", href):
            continue
        full = urljoin(AC_PAGE, href)
        if full in seen:
            continue
        seen.add(full)
        # 从文件名推产品名
        stem = Path(href).stem
        nm = re.sub(r"[_—-]*(条款|更新).*$", "", stem).strip()
        nm = re.sub(r"_+\d*$", "", nm).strip()
        items.append({
            "insurer": "中英人寿",
            "name": nm,
            "url": full,
            "doc_no": "",
            "reg_no": "",
            "page": AC_PAGE,
            "kind": guess_kind(nm),
        })
    log(f"解析到 {len(items)} 个条款 PDF（已滤掉产品说明书 + 历史版本）")
    for it in items:
        log(f"  [{it['kind']}] {it['name']}  <- {it['url'].rsplit('/',1)[-1]}")
    return items


# ----------------------------------------------------------------------
# 3. 下载 + 体检（文本层 / 页数）
# ----------------------------------------------------------------------
def safe_name(s: str) -> str:
    """Windows 文件名净化：去掉路径分隔符与非法字符，全角括号保留。"""
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "", s)
    s = re.sub(r"\s+", "", s)
    return s[:90]


def download(items):
    log("")
    log("=" * 72)
    log("### 3. 下载 + 体检")
    rows, idx = [], 0
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
        log("!! 未装 PyMuPDF，跳过文本层体检")

    for it in items:
        idx += 1
        code = "CPIC" if it["insurer"] == "太平洋人寿" else "AC"
        fname = f"{idx:02d}_{code}_{it['kind']}_{safe_name(it['name'])}.pdf"
        fpath = PDF_DIR / fname

        if fpath.exists() and fpath.stat().st_size > 1000:
            log(f"[{idx:02d}] 已存在，跳过：{fname}")
        else:
            try:
                # 中文 URL 需编码；已编码的保持原样
                u = quote(it["url"], safe=":/?&=#%+@$,;~()!'*[]")
                rr = requests.get(u, headers={**BASE_H, "Referer": it["page"]},
                                  timeout=90, verify=False)
                if rr.status_code != 200 or not rr.content.startswith(b"%PDF"):
                    log(f"[{idx:02d}] **下载失败** status={rr.status_code} "
                        f"magic={rr.content[:8]!r}  {it['name']}")
                    continue
                fpath.write_bytes(rr.content)
                size = len(rr.content)
                log(f"[{idx:02d}] OK  {size/1024:8.1f} KB  {fname}")
                time.sleep(1.0)  # 限速，做有礼貌的爬虫
            except Exception as e:
                log(f"[{idx:02d}] **异常** {type(e).__name__}: {e}  {it['name']}")
                continue

        # 体检
        pages, text_chars, sample = "", "", ""
        if fitz:
            try:
                d = fitz.open(str(fpath))
                pages = d.page_count
                txt = "".join(d[i].get_text() for i in range(min(pages, 3)))
                text_chars = len(txt.strip())
                sample = re.sub(r"\s+", " ", txt.strip())[:60]
                d.close()
            except Exception as e:
                sample = f"<fitz error {e}>"

        blob = fpath.read_bytes()
        rows.append({
            "序号": idx, "保司": it["insurer"], "险种": it["kind"],
            "产品名称": it["name"], "文件名": fname,
            "报备文件编号": it["doc_no"], "产品注册号": it["reg_no"],
            "页数": pages, "前3页文本字符数": text_chars,
            "字节数": len(blob),
            "SHA256": hashlib.sha256(blob).hexdigest()[:16],
            "来源页面": it["page"], "原始URL": it["url"],
            "下载时间": time.strftime("%Y-%m-%d %H:%M"),
            "正文样例": sample,
        })

    return rows


def write_csv(rows):
    cols = ["序号", "保司", "险种", "产品名称", "文件名", "报备文件编号",
            "产品注册号", "页数", "前3页文本字符数", "字节数", "SHA256",
            "来源页面", "原始URL", "下载时间", "正文样例"]
    with io.open(SRC_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log("")
    log(f"sources.csv 写入 {len(rows)} 行 -> {SRC_CSV}")


if __name__ == "__main__":
    check_mineru()
    items = fetch_cpic() + fetch_aviva()
    log("")
    log(f"### 合计待采 {len(items)} 份")
    rows = download(items)
    write_csv(rows)

    log("")
    log("=" * 72)
    log("### 4. 汇总")
    tot = sum(r["字节数"] for r in rows)
    log(f"成功落盘 {len(rows)} 份，合计 {tot/1024/1024:.1f} MB")
    kk = {}
    for r in rows:
        kk[r["险种"]] = kk.get(r["险种"], 0) + 1
    log("险种分布：" + " / ".join(f"{k}×{v}" for k, v in kk.items()))
    scan = [r["文件名"] for r in rows if isinstance(r["前3页文本字符数"], int)
            and r["前3页文本字符数"] < 20]
    log(f"疑似扫描件（前3页无文本层）：{scan if scan else '无'}")

    txt = "\n".join(LOG)
    (ROOT / "data" / "clauses" / "_collect_log.txt").write_text(txt, encoding="utf-8")
    print(txt)
