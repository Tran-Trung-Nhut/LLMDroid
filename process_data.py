import pandas as pd

def main():
    # 1. Đọc file và chỉ lấy 2 cột pkg_name, Result
    df = pd.read_csv('data/apps_raw.csv', usecols=['pkg_name', 'Result'])
    
    # 2. Đổi tên cột Result thành label
    df = df.rename(columns={'Result': 'label'})
    
    # 3. Xóa các dòng có label là "Can't download"
    df = df[df['label'] != "Can't download"]
    
    # Ánh xạ các giá trị "Not found", "AI Only" thành 0 và "Found" thành 1
    mapping = {
        'Not found': 0,
        'AI Only': 0,
        'Found': 1
    }
    
    # Map giá trị
    df['label'] = df['label'].map(mapping)
    
    # Loại bỏ các dòng bị rỗng (NaN) ở cột label nếu có
    df = df.dropna(subset=['label'])
    
    # Đổi kiểu dữ liệu sang int cho đẹp (0, 1 thay vì 0.0, 1.0)
    df['label'] = df['label'].astype(int)
    
    # 4. Xuất ra file inference_manual.csv
    df.to_csv('inference_manual.csv', index=False)
    print("Đã tạo file inference_manual.csv")
    
    # 5. Tạo file inference_apps.csv chỉ với cột pkg_name
    df[['pkg_name']].to_csv('inference_apps.csv', index=False)
    print("Đã tạo file inference_apps.csv")

if __name__ == '__main__':
    main()
