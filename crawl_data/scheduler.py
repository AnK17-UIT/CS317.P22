import schedule
import time
from pipeline import run_pipeline, export_to_csv

def job():
    print("▶️ Đang chạy pipeline...")
    cases = run_pipeline(pages=30)
    export_to_csv(cases)
    print("✅ Xong!")

# Chạy lúc 07:00 mỗi ngày
schedule.every().day.at("07:00").do(job)

print("📅 Đã thiết lập lịch tự động.")
while True:
    schedule.run_pending()
    time.sleep(60)
