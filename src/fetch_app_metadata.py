#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from google_play_scraper import app as gplay_app
from tqdm import tqdm

# Add src to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


class AppMetadataFetcher:    
    def __init__(self):
        self.images_dir = Path(CFG.images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_play_store_metadata(self, pkg_name: str) -> Optional[Dict]:
        try:
            result = gplay_app(
                pkg_name,
                lang=CFG.gplay_lang,
                country=CFG.gplay_country
            )
            return result
        except Exception:
            return None
    
    def download_screenshots(self, pkg_name: str, screenshot_urls: List[str]) -> List[str]:
        app_image_dir = self.images_dir / pkg_name
        app_image_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        
        for idx, url in enumerate(screenshot_urls, start=1):
            image_path = app_image_dir / f"{idx}.{CFG.image_format}"
    
            if image_path.exists():
                rel_path = f"{CFG.images_dir}/{pkg_name}/{idx}.{CFG.image_format}"
                image_paths.append(rel_path)
                continue

            try:
                response = requests.get(url, timeout=CFG.screenshot_download_timeout)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    
                    rel_path = f"{CFG.images_dir}/{pkg_name}/{idx}.{CFG.image_format}"
                    image_paths.append(rel_path)
                    
            except Exception:
                continue
        
        return image_paths
    
    def process_app(self, pkg_name: str) -> Optional[Dict]:
        # Fetch Play Store metadata
        play_data = self.fetch_play_store_metadata(pkg_name)
        if not play_data:
            return None
        
        # Get recent changes directly from Play Store
        recent_changes = play_data.get('recentChanges', '')
        
        # Download screenshots
        screenshot_urls = play_data.get('screenshots', [])
        image_paths = self.download_screenshots(pkg_name, screenshot_urls)
        
        # Format data to match apps_raw.jsonl structure
        formatted_data = {
            'app_id': pkg_name,
            'label_binary': None, 
            'label_3class': None,
            'title': play_data.get('title', ''),
            'description': play_data.get('description', ''),
            'short_description': play_data.get('summary', ''),
            'recent_changes_text': recent_changes if recent_changes else '',
            'category': play_data.get('genre', ''),
            'image_paths': image_paths
        }
        
        return formatted_data
    
    def process_apps_csv(self, input_csv: str, output_jsonl: str):
        df = pd.read_csv(input_csv)
        pkg_names = df['pkg_name'].dropna().unique()
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Found {len(pkg_names)} unique package names")
        print(f"Writing output to: {output_jsonl}")
        
        results = []
        failed = []
        
        for pkg_name in tqdm(pkg_names, desc="Processing apps", unit="app"):
            try:
                result = self.process_app(pkg_name)
                if result:
                    results.append(result)
                else:
                    failed.append(pkg_name)
                
                time.sleep(CFG.api_request_delay)
                
            except Exception:
                failed.append(pkg_name)
                continue
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\nProcessed: {len(results)} apps")
        print(f"Failed: {len(failed)} apps")
        print(f"Output saved to: {output_jsonl}")
        print(f"Images saved to: {self.images_dir}")
        
        if failed:
            print(f"\nFailed apps:")
            for pkg in failed[:CFG.max_failed_apps_display]:
                print(f"  - {pkg}")
            if len(failed) > CFG.max_failed_apps_display:
                print(f"  ... and {len(failed) - CFG.max_failed_apps_display} more")
            
            failed_file = Path(output_jsonl).parent / CFG.failed_apps_filename
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed))
            print(f"\nFull list of failed apps saved to: {failed_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch app metadata directly from Google Play Store'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file with pkg_name column'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file'
    )
    
    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    
    fetcher = AppMetadataFetcher()
    fetcher.process_apps_csv(args.input, args.output)
    
    print("\n✓ Done! You can now run the inference pipeline.")


if __name__ == '__main__':
    main()