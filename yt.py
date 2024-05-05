import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
import pickle

# https://developers.google.com/youtube/v3/guides/uploading_a_video

scopes = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


def authenticate_youtube():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    client_secrets_file = "cache/yt-clientsecret.json"

    if os.path.exists("cache/yt.token.pickle"):
        with open("cache/yt.token.pickle", "rb") as token:
            credentials = pickle.load(token)
    else:
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
        credentials = flow.run_local_server(port=8080)
        with open("cache/yt.token.pickle", "wb") as token:
            pickle.dump(credentials, token)

    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
    return youtube


def upload_video(youtube, file_name, title, description, category_id, keywords):
    request_body = {
        "snippet": {"category_id": category_id, "title": title, "description": description, "tags": keywords},
        "status": {"privacyStatus": "unlisted"},
    }

    media_file = MediaFileUpload(file_name, mimetype="video/mp4", resumable=True, chunksize=1024 * 1024)
    response_upload = youtube.videos().insert(
        part=",".join(request_body.keys()), body=request_body, media_body=media_file
    )

    print(response_upload)

    response = None
    while response is None:
        status, response = response_upload.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%.")

    print(f"Video ID: {response.get('id')}")

    return response.get("id")


def add_to_playlist(youtube, video_id, playlist_id):
    playlist_request_body = {
        "snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": video_id}}
    }

    youtube.playlistItems().insert(part="snippet", body=playlist_request_body).execute()


def yt():
    youtube = authenticate_youtube()
    video_id = upload_video(
        youtube,
        "/workspaces/ArxivPapers/.temp/1706.03762_short.mp4",
        "1706.03762_short",
        "Video Description",
        "24",
        None,
    )
    add_to_playlist(youtube, video_id, "PL9PmEzz6RJfJcFZ8Jft1vJ9e8wVgJCrGp")


# yt()
