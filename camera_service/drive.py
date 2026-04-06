import os
import time
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

# OAuth 2.0 client secrets file downloaded from Google Cloud Console
CLIENT_SECRETS_FILE = 'client_secret_666956225711-odlet1oft281i8dpbg2ckgt6c422qqde.apps.googleusercontent.com.json'  # Replace with your OAuth 2.0 JSON file

SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

# ID of the spreadsheet where links will be recorded
SPREADSHEET_ID = '1dT3KXtuPwyFEuZVWcEuaRP5nZmFSFF4eYzhGtEE73vc'
# The parent folder ID to create folders inside (use 'root' for My Drive root)
PARENT_FOLDER_ID = 'root'

def get_credentials():
    creds = None
    # token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If no valid credentials, let user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def create_folder(service_drive, name, parent_id):
    file_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service_drive.files().create(body=file_metadata, fields='id, webViewLink').execute()
    return folder['id'], folder['webViewLink']

def set_folder_permission(service_drive, folder_id):
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service_drive.permissions().create(
        fileId=folder_id,
        body=permission,
        fields='id'
    ).execute()

def append_to_sheet(service_sheets, sheet_id, values):
    body = {'values': values}
    service_sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range='Sheet1!A:B',
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body=body
    ).execute()

def main():
    creds = get_credentials()
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    batch_size = 50  # batch appending to reduce API calls
    batch_values = []
    total_folders = 2000

    for i in range(1, total_folders + 1):
        folder_name = f'Folder_{i}'
        try:
            folder_id, folder_link = create_folder(drive_service, folder_name, PARENT_FOLDER_ID)
            set_folder_permission(drive_service, folder_id)
            batch_values.append([i, folder_link])
            print(f"Created {folder_name}: {folder_link}")
        except HttpError as error:
            print(f"An error occurred: {error}")
            time.sleep(5)

        if len(batch_values) >= batch_size:
            append_to_sheet(sheets_service, SPREADSHEET_ID, batch_values)
            batch_values = []

    # Append remaining values
    if batch_values:
        append_to_sheet(sheets_service, SPREADSHEET_ID, batch_values)

if __name__ == '__main__':
    main()
