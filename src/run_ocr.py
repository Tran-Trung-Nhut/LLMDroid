import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.config import CFG
import pytesseract


def run_ocr_on_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang='eng')
        return text.strip()
    except Exception as e:
        return ""


def main():    
    # Read dataset
    rows = []
    with open(CFG.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    
    total_images = sum(len(r.get('image_paths', [])) for r in rows)
    print(f"Found {len(rows)} apps with {total_images} images")
    
    # Process each app
    processed = 0
    pbar = tqdm(total=total_images, desc="Running OCR")
    
    for row in rows:
        image_paths = row.get('image_paths', [])
        if not image_paths:
            continue
        
        if 'ocr_by_image' not in row or not isinstance(row['ocr_by_image'], dict):
            row['ocr_by_image'] = {}
        
        for img_path in image_paths:
            if img_path in row['ocr_by_image']:
                pbar.update(1)
                continue
            
            ocr_text = run_ocr_on_image(img_path)
            row['ocr_by_image'][img_path] = ocr_text
            processed += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save results and show statistics
    if processed > 0:
        print(f"Saving {processed} new OCR results...")
        with open(CFG.dataset_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        # Count how many OCR texts are non-empty
        total_ocr = 0
        non_empty_ocr = 0
        for row in rows:
            for text in row.get('ocr_by_image', {}).values():
                total_ocr += 1
                if text and len(text.strip()) > 0:
                    non_empty_ocr += 1
        
        print("✓ OCR completed")
        print(f"  Total: {total_ocr} images")
        print(f"  With text: {non_empty_ocr} ({non_empty_ocr/total_ocr*100:.1f}%)")
        print(f"  Empty: {total_ocr - non_empty_ocr} ({(total_ocr-non_empty_ocr)/total_ocr*100:.1f}%)")
        
        if non_empty_ocr == 0:
            print("\n⚠️  WARNING: No text extracted from any images!")
            print("   Possible reasons:")
            print("   - Images have no text")
            print("   - Images are too blurry")
            print("   - Tesseract not working properly")
            print("   → Model will use visual-only features (middle image selection)")
    else:
        print("All images already have OCR data")


if __name__ == "__main__":
    main()
