import streamlit as st
from PIL import Image, ImageDraw
from ocr import SuryaClient
import fitz
import io

ZOOM_FACTOR = 1.0
MERGE_GAP_PX = 5
MIN_BLOCK_AREA = 100

def merge_blocks(blocks, merge_gap_px):
    blocks.sort(key=lambda b: b['bbox'][1])
    merged = []
    group = [blocks[0]]

    for block in blocks[1:]:
        last = group[-1]
        gap = block['bbox'][1] - last['bbox'][3]
        if gap < merge_gap_px:
            group.append(block)
        else:
            merged.append(merge_group(group))
            group = [block]

    if group:
        merged.append(merge_group(group))

    return merged

def merge_group(group):
    min_x = min(b['bbox'][0] for b in group)
    min_y = min(b['bbox'][1] for b in group)
    max_x = max(b['bbox'][2] for b in group)
    max_y = max(b['bbox'][3] for b in group)
    return {
        "bbox": [min_x, min_y, max_x, max_y],
        "area": (max_x - min_x) * (max_y - min_y)
    }

def main():
    st.title("Surya GUI – blokowy layout z mergem")

    uploaded_file = st.file_uploader("Wybierz PDF lub obraz", type=["pdf", "png", "jpg", "jpeg"])
    if not uploaded_file:
        return

    images = []
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM_FACTOR, ZOOM_FACTOR))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
        doc.close()
    else:
        img = Image.open(uploaded_file).convert("RGB")
        images.append(img)

    client = SuryaClient()
    results = client.process_pages(images)

    for i, (img, result) in enumerate(zip(images, results)):
        draw = ImageDraw.Draw(img)
        raw_blocks = []
        layout = result.get("layout")

        if hasattr(layout, "bboxes"):
            layout = layout.bboxes

        for elem in layout:
            bbox = elem.bbox if hasattr(elem, 'bbox') else elem.get("bbox", [])
            if len(bbox) != 4:
                continue
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            area = w * h
            raw_blocks.append({"bbox": bbox, "area": area})

        merged = merge_blocks(raw_blocks, MERGE_GAP_PX)
        big_blocks = [b for b in merged if b['area'] > MIN_BLOCK_AREA]

        for b in big_blocks:
            draw.rectangle(b['bbox'], outline="red", width=2)

        st.image(img, caption=f"Strona {i + 1} ({len(big_blocks)} bloków)")

if __name__ == "__main__":
    main()
