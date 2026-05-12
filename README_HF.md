---
title: ViSum
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Hệ thống Tóm tắt Báo chí Tiếng Việt
license: mit
---

# 🇻🇳 ViSum — Tóm tắt Báo chí Tiếng Việt

Hệ thống tóm tắt văn bản tiếng Việt sử dụng BARTpho + QLoRA.

## Hướng dẫn
- Dán link bài báo (VnExpress, Tuổi Trẻ, Thanh Niên...) hoặc
- Dán thẳng nội dung văn bản vào ô bên dưới
- Nhấn **Tóm tắt ngay**

## Công nghệ
- Model: OrdinaryAI/visum-qlora-5epochs (BARTpho + QLoRA 5 epochs)
- Dataset: harouzie/vietnews (144K mẫu)
- Frontend: Gradio
- Deploy: Hugging Face Spaces
