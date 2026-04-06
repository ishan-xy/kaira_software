import os
import re
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
from urllib.parse import urlparse, parse_qs


# ========== CONFIG ==========
FOLDER_PATH = r"C:\Users\SHAURYA\OneDrive\Desktop\KAIRA\camera_service\signed_photos"
EXCEL_FILE = "Satpic.xlsx"  # Local Excel file
CREDENTIALS_FILE = "cedar-scene-477016-a0-51fe754b2e03.json"


# ========== AUTH ==========
SCOPES = [
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)


# ========== READ EXCEL FILE ==========
try:
    df = pd.read_excel(EXCEL_FILE, sheet_name="Sheet1")
    print(f"✅ Loaded {EXCEL_FILE}")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"❌ Excel file not found: {EXCEL_FILE}")
    exit(1)


# Create mapping from photo number to drive folder link
mapping = {}
for _, row in df.iterrows():
    try:
        num = str(int(row.iloc[0]))  # First column = photo number
        link = str(row.iloc[1])       # Second column = Drive folder link
        mapping[num] = link
    except (ValueError, IndexError):
        continue

print(f"📋 Found {len(mapping)} photo-to-folder mappings in Excel")


# ========== EXTRACT FOLDER ID FROM URL ==========
def extract_folder_id(drive_url):
    """Extract folder ID from Google Drive URL"""
    parsed = urlparse(drive_url)
    
    # Handle https://drive.google.com/drive/folders/XXXXX format
    if "folders" in parsed.path:
        return parsed.path.split("/folders/")[-1].strip("/")
    
    # Handle https://drive.google.com/open?id=XXXXX format
    query_params = parse_qs(parsed.query)
    if "id" in query_params:
        return query_params["id"][0]
    
    # Fallback: try regex
    match = re.search(r'[-\w]{25,}', drive_url)
    if match:
        return match.group(0)
    
    return None


# ========== UPLOAD FILES ==========
uploaded_count = 0
failed_count = 0

for filename in os.listdir(FOLDER_PATH):
    # Match files like "1_signed.png", "2_signed.png"
    match = re.match(r"(\d+)_signed\.png", filename)
    if not match:
        continue

    number = match.group(1)
    folder_link = mapping.get(number)

    if not folder_link:
        print(f"⚠️  No mapping found in Excel for image #{number} ({filename})")
        continue

    # Extract folder ID from link
    folder_id = extract_folder_id(folder_link)
    if not folder_id:
        print(f"❌ Invalid folder link in Excel for {filename}: {folder_link}")
        failed_count += 1
        continue

    file_path = os.path.join(FOLDER_PATH, filename)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        failed_count += 1
        continue

    # Upload to Drive
    file_metadata = {'name': filename, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, resumable=True)

    try:
        uploaded = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
        print(f"✅ Uploaded {filename} → {uploaded['webViewLink']}")
        uploaded_count += 1
    except Exception as e:
        print(f"❌ Failed to upload {filename}: {e}")
        failed_count += 1


print(f"\n📊 Summary: {uploaded_count} uploaded, {failed_count} failed")
