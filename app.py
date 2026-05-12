# ============================================================
# ViSum — Hệ thống Tóm tắt Báo chí Tiếng Việt
# Gradio UI | Crawl link báo | Trích nguồn | Beam cố định=4
# ============================================================

import re
import time
import torch
import gradio as gr
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# ============================================================
# CONFIG
# ============================================================
MODEL_ID  = "OrdinaryAI/visum-qlora-5epochs"
NUM_BEAMS = 4  # Cố định beam=4: cân bằng tốc độ & chất lượng trên CPU

# ============================================================
# MODEL LOADING
# ============================================================
_tokenizer = None
_model     = None

def get_model():
    global _tokenizer, _model
    if _model is None:
        peft_config = PeftConfig.from_pretrained(MODEL_ID)
        base_model  = AutoModelForSeq2SeqLM.from_pretrained(
            peft_config.base_model_name_or_path
        )
        model = PeftModel.from_pretrained(base_model, MODEL_ID)
        model = model.merge_and_unload()
        _tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = model.to(device)
        _model.eval()
    return _tokenizer, _model


# ============================================================
# FIX DÍNH TỪ (BARTpho syllable artifact)
# ============================================================
def fix_bartpho_output(text: str) -> str:
    # Fix chữ hoa dính chữ thường
    text = re.sub(
        r'([a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])'
        r'([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])',
        r'\1 \2', text
    )
    # Fix dấu câu dính chữ
    text = re.sub(r'([.!?,;:])([^\s\d"»)])', r'\1 \2', text)
    # Fix ngoặc dính chữ: "(Hà" → "( Hà", "Nội)" → "Nội )"
    text = re.sub(r'\(([^\s])', r'( \1', text)
    text = re.sub(r'([^\s])\)', r'\1 )', text)
    # Xóa space thừa
    text = re.sub(r' +', ' ', text).strip()
    # Fix lại ngoặc cho đẹp
    text = re.sub(r'\( ', '(', text)
    text = re.sub(r' \)', ')', text)
    return text


# ============================================================
# CLEAN INPUT TEXT
# ============================================================
def clean_input_text(text: str) -> str:
    # Xóa ký tự unicode rác
    text = text.replace("\u200b", " ").replace("\ufeff", " ").replace("\u00a0", " ")
    text = re.sub(r"\t+", " ", text)

    # Fix dính chữ kiểu: thànhphố, côngan, vụviệc...
    # Dùng wordninja nếu có, không thì dùng regex
    # Regex: thêm space giữa chữ thường+hoa (ViệtNam → Việt Nam)
    text = re.sub(
        r'([a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])'
        r'([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])',
        r'\1 \2', text
    )

    # Fix dấu câu dính chữ: "vong.Sau" → "vong. Sau"
    text = re.sub(r'([.!?,;:])([^\s\d"»])', r'\1 \2', text)

    # Xóa space/newline thừa
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CRAWL BÀI BÁO
# ============================================================
NEWSPAPER_MAP = {
    "vnexpress.net":  "VnExpress",
    "tuoitre.vn":     "Tuổi Trẻ",
    "thanhnien.vn":   "Thanh Niên",
    "dantri.com.vn":  "Dân Trí",
    "nhandan.vn":     "Nhân Dân",
    "vietnamnet.vn":  "VietnamNet",
    "baomoi.com":     "Báo Mới",
    "zing.vn":        "Zing News",
    "cafef.vn":       "CafeF",
    "laodong.vn":     "Lao Động",
    "tienphong.vn":   "Tiền Phong",
    "vtv.vn":         "VTV",
    "baochinhphu.vn": "Báo Chính Phủ",
    "sggp.org.vn":    "Sài Gòn Giải Phóng",
}

def get_newspaper_name(url: str) -> str:
    domain = urlparse(url).netloc.replace("www.", "")
    for key, name in NEWSPAPER_MAP.items():
        if key in domain:
            return name
    return domain

def crawl_article(url: str) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "figure", "figcaption", "iframe",
                          "noscript", "form", "button", "img"]):
            tag.decompose()
        for cls in [".advertisement", ".ads", ".related", ".sidebar", ".comment"]:
            for el in soup.select(cls):
                el.decompose()

        title = ""
        for sel in ["h1.title-detail", "h1.article-title", "h1.title",
                    "h1", 'meta[property="og:title"]']:
            el = soup.select_one(sel)
            if el:
                title = el.get("content", "") or el.get_text(strip=True)
                if title:
                    break

        author = ""
        for sel in [".author", ".author-name", '[rel="author"]',
                    ".article-author", ".byline", ".reporter",
                    'meta[name="author"]']:
            el = soup.select_one(sel)
            if el:
                author = el.get("content", "") or el.get_text(strip=True)
                if author:
                    break

        content = ""
        for sel in [
            "article.fck_detail", ".article-body", ".article-content",
            ".content-detail", "#article-body", "div.fck_detail",
            "div.singular-content", "div[itemprop='articleBody']",
            "section.article", ".post-content", "main article",
        ]:
            el = soup.select_one(sel)
            if el:
                paragraphs = el.find_all("p")
                if paragraphs:
                    content = "\n".join(
                        p.get_text(strip=True) for p in paragraphs
                        if len(p.get_text(strip=True)) > 30
                    )
                    if len(content) > 200:
                        break

        if not content:
            paragraphs = soup.find_all("p")
            content = "\n".join(
                p.get_text(strip=True) for p in paragraphs
                if len(p.get_text(strip=True)) > 30
            )

        return {
            "content":   content.strip(),
            "title":     title,
            "author":    author,
            "newspaper": get_newspaper_name(url),
            "url":       url,
        }
    except Exception as e:
        return {
            "content": "", "title": "", "author": "",
            "newspaper": get_newspaper_name(url),
            "url": url, "error": str(e),
        }


# ============================================================
# TÓM TẮT
# Không dùng min_length — để model tự quyết khi nào dừng
# Ép min_length là phi logic: bản tóm tắt không có độ dài tối thiểu
# ============================================================
def summarize(text: str, max_length: int) -> dict:
    tokenizer, model = get_model()
    device = next(model.parameters()).device

    start = time.time()
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=1024, truncation=True, padding=True,
    ).to(device)

    input_len = len(inputs["input_ids"][0])

    # smart_max: tự điều chỉnh theo độ dài bài, không vượt 500
    smart_max = max(max_length, int(input_len * 0.4))
    smart_max = min(smart_max, 700)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=smart_max,
            min_length=min(120, int(input_len * 0.08)),
            # Không có min_length — model tự dừng khi đủ ý
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            length_penalty=0.9,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary = fix_bartpho_output(summary)
    return {"summary": summary, "time": round(time.time() - start, 2)}


# ============================================================
# HÀM XỬ LÝ CHÍNH
# ============================================================
def process(url_input: str, text_input: str, max_length: int):
    source_info = ""

    if url_input.strip():
        yield "⏳ Đang tải nội dung bài báo...", "", ""
        article_data = crawl_article(url_input.strip())
        if not article_data.get("content"):
            err = article_data.get("error", "Không rõ nguyên nhân")
            yield f"❌ Không crawl được.\nLỗi: {err}\n\nThử dán nội dung thủ công.", "", ""
            return
        text = article_data["content"]
        parts = []
        if article_data.get("title"):
            parts.append(f"Bài báo **{article_data['title']}**")
        if article_data.get("newspaper"):
            parts.append(f"Tờ báo: {article_data['newspaper']}")
        if article_data.get("author"):
            parts.append(f"Tác giả: {article_data['author']}")
        parts.append(f"🔗 Nguồn: {url_input.strip()}")
        source_info = "\n\n".join(parts)

    elif text_input.strip():
        text = clean_input_text(text_input)
        source_info = "📝 Văn bản nhập trực tiếp"
    else:
        yield "⚠️ Vui lòng nhập link báo hoặc dán văn bản!", "", ""
        return

    if len(text.split()) < 80:
        yield "⚠️ Nội dung quá ngắn (cần ít nhất 80 từ).", source_info, ""
        return

    yield "Đang tóm tắt... Vui lòng chờ trong giây lát.", source_info, ""

    result     = summarize(text, max_length)
    summary    = result["summary"]
    elapsed    = result["time"]
    orig_words = len(text.split())
    summ_words = len(summary.split())
    reduction  = round((1 - summ_words / orig_words) * 100, 1)

    stats = (
        f"⏱️ {elapsed}s  |  Gốc: {orig_words} từ  |  "
        f"Tóm tắt: {summ_words} từ  |  Rút gọn: {reduction}%"
    )
    yield summary, source_info, stats


# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Be+Vietnam+Pro:wght@300;400;500;600&display=swap');

:root {
    --red-dark:   #8B0000;
    --gold:       #FFCD00;
    --gold-light: #FFE566;
    --cream:      #FFF8E7;
    --text-light: #FFF5E0;
}

body, .gradio-container {
    background: linear-gradient(160deg, #5C0011 0%, #8B0000 40%, #A00020 70%, #6B0010 100%) !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    min-height: 100vh;
}

.visum-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 30px 10px;
    border-bottom: 2px solid rgba(255,205,0,0.4);
    margin-bottom: 24px;
}

.visum-title-group {
    display: flex;
    align-items: center;
    gap: 16px;
}

.visum-crane-wrap {
    width: 70px;
    height: 70px;
    animation: float 3s ease-in-out infinite, glowPulse 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(255,205,0,0.7));
}

.visum-crane-svg { width: 100%; height: 100%; }

@keyframes float {
    0%, 100% { transform: translateY(0px)   rotate(-2deg); }
    25%       { transform: translateY(-9px)  rotate(0deg);  }
    50%       { transform: translateY(-5px)  rotate(-3deg); }
    75%       { transform: translateY(-11px) rotate(-1deg); }
}

@keyframes glowPulse {
    0%, 100% { filter: drop-shadow(0 0 8px  rgba(255,205,0,0.5));  }
    50%       { filter: drop-shadow(0 0 18px rgba(255,215,0,0.95)); }
}

.visum-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: var(--gold) !important;
    text-shadow: 0 0 20px rgba(255,205,0,0.5), 2px 2px 4px rgba(0,0,0,0.5);
    line-height: 1.1;
    margin: 0;
}

.visum-subtitle {
    color: var(--text-light);
    font-size: 0.95rem;
    opacity: 0.85;
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 0.5px;
}

.visum-flag {
    font-size: 64px;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));
}

.panel-card {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,205,0,0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(8px);
}

label, .label-wrap span {
    color: var(--gold-light) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

input[type="text"], textarea {
    background: rgba(0,0,0,0.35) !important;
    border: 1.5px solid rgba(255,205,0,0.3) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(255,205,0,0.15) !important;
}

input[type="text"]::placeholder, textarea::placeholder {
    color: rgba(255,245,224,0.4) !important;
}

.gr-button-primary, button.primary {
    background: linear-gradient(135deg, var(--gold) 0%, #E6B800 100%) !important;
    color: var(--red-dark) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 15px rgba(255,205,0,0.3) !important;
    transition: all 0.2s ease !important;
}

.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255,205,0,0.5) !important;
    background: linear-gradient(135deg, #FFE566 0%, var(--gold) 100%) !important;
}

input[type="range"] { accent-color: var(--gold) !important; }

.output-box {
    background: rgba(0,0,0,0.4) !important;
    border: 1.5px solid rgba(255,205,0,0.35) !important;
    border-radius: 12px !important;
    color: var(--cream) !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
}

.source-box {
    background: rgba(255,205,0,0.08) !important;
    border: 1px solid rgba(255,205,0,0.3) !important;
    border-radius: 10px !important;
    color: var(--gold-light) !important;
    font-size: 0.9rem !important;
}

.stats-box {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid rgba(255,205,0,0.2) !important;
    border-radius: 8px !important;
    color: var(--text-light) !important;
    font-size: 0.85rem !important;
    text-align: center !important;
}

.section-header {
    color: var(--gold) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
}

.gr-accordion {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,205,0,0.2) !important;
    border-radius: 10px !important;
}

.visum-footer {
    background: linear-gradient(90deg, var(--red-dark) 0%, var(--gold) 50%, var(--red-dark) 100%);
    text-align: center;
    padding: 14px 20px;
    margin-top: 32px;
    border-radius: 12px;
    color: var(--red-dark);
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}

.gr-form { gap: 14px !important; }
"""

# ============================================================
# HEADER HTML 
# ============================================================
HEADER_HTML = """
<div class="visum-header">
    <div class="visum-title-group">

        <div class="visum-crane-wrap">
            <svg class="visum-crane-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <polygon
                    points="50,5 61,35 95,35 68,57 79,91 50,70 21,91 32,57 5,35 39,35"
                    fill="#FFCD00"
                    stroke="#E6A800"
                    stroke-width="1.5"
                    stroke-linejoin="round"
                />
            </svg>
        </div>
        <div>
            <div class="visum-title">ViSum</div>
            <div class="visum-subtitle">Hệ thống Tóm tắt Báo chí Tiếng Việt • AI-Powered</div>
        </div>
    </div>
    <div class="visum-flag">🇻🇳</div>
</div>
"""

FOOTER_HTML = """
<div class="visum-footer">
    🇻🇳 ViSum • Vietnamese News Summarization •
    Powered by BARTpho + QLoRA • Ordinary-AI-Engineer
</div>
"""

# ============================================================
# GRADIO APP
# ============================================================
with gr.Blocks(css=CUSTOM_CSS, title="ViSum — Tóm tắt Báo chí Tiếng Việt") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        # CỘT TRÁI — INPUT
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("Nhập liệu", elem_classes="section-header")

            url_input = gr.Textbox(
                label="🔗 Link bài báo (VnExpress, Tuổi Trẻ, Thanh Niên...)",
                placeholder="https://vnexpress.net/...",
                lines=1,
            )
            gr.Markdown(
                "<center style='color:rgba(255,245,224,0.45);font-size:0.85rem;margin:4px 0'>— hoặc —</center>"
            )
            text_input = gr.Textbox(
                label="📋 Dán văn bản trực tiếp",
                placeholder="Dán nội dung bài báo tiếng Việt vào đây...",
                lines=10,
            )

            with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                gr.Markdown(
                    "<small style='color:rgba(255,245,224,0.6)'>"
                    "Beam Search cố định ở <b style='color:#FFCD00'>4</b> "
                    "— cân bằng tối ưu giữa tốc độ và chất lượng trên CPU.<br>"
                    "Chỉ điều chỉnh độ dài tối đa nếu muốn tóm tắt ngắn hơn hoặc dài hơn.</small>"
                )
                max_length = gr.Slider(
                    label="Độ dài tóm tắt tối đa (token)",
                    minimum=80, maximum=500, value=250, step=10,
                )

            submit_btn = gr.Button("Tóm tắt ngay", variant="primary", size="lg")

        # CỘT PHẢI — OUTPUT
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("Kết quả", elem_classes="section-header")

            source_output = gr.Markdown(
                label="Thông tin nguồn",
                elem_classes="source-box",
                value="",
            )
            summary_output = gr.Textbox(
                label="Bản tóm tắt",
                lines=12,
                interactive=False,
                elem_classes="output-box",
                placeholder="Bản tóm tắt sẽ xuất hiện ở đây...",
            )
            stats_output = gr.Textbox(
                label="Thống kê",
                interactive=False,
                elem_classes="stats-box",
                placeholder="Thời gian | Số từ | Tỉ lệ rút gọn",
            )

    gr.HTML(FOOTER_HTML)

    # Sự kiện
    submit_btn.click(
        fn=process,
        inputs=[url_input, text_input, max_length],
        outputs=[summary_output, source_output, stats_output],
    )
    url_input.submit(
        fn=process,
        inputs=[url_input, text_input, max_length],
        outputs=[summary_output, source_output, stats_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, ssr_mode=False)
