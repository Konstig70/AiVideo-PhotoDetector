from flask import Flask, request, jsonify 
from pymediainfo import MediaInfo
from flask_cors import CORS
from MetaDataScrutiny.metadataanalyzer import metadata
from GeometryMapping.GeometryMapping import GeometryMapper
from Justification.justification import VideoJustificationAgent
from Justification.justification import PDFGenerator
import uuid
import os
from google import genai
import tempfile


app = Flask(__name__)
CORS(app)


@app.route("/", methods=['POST'])
def hello_world():
    #First get video and download to a temp
    file = request.files.get("file")
    response = "Default response, this means that no operations were done"
    if file:
        response = saveFile(file)
        
    yt_url = request.form.get("youtube_url")
    #Get path to tempfile
    path = request.form.get("path");
    #Get function and match
    fun = request.form.get("function")
    prompt = request.form.get("prompt")
    print(file)
    print(f"function passed was {fun}")
    match fun:
        case "metadata":
            response = performMetadata(path)
        case "anatomy":
            response = performGeometryAnalysis(path)
            if path:
                os.remove(path)

        case "video_data":
            response = getVideoData(path)
        case "analysis":
            response = getAgentResponse(prompt)
        
    return response 

def getAgentResponse(prompt):
    print("Getting agent response")
    with open("apiavain.txt") as f:
        print("Opened file")
        client = genai.Client(api_key=f.read())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text

def saveFile(file):
    #Save video locally
    ext = file.filename.split('.')[-1]
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", dir="uploads")
    file.save(temp.name)

    #Update the address so that we read the correct data 
    return temp.name

def performMetadata(path):
    
    print("Performing metadata analysis")
    analyzer = metadata(path)
    return analyzer.extract_metadata()

def performGeometryAnalysis(path): 
    print("Performing geometrical analysis")
    anatomy_analyser = GeometryMapper()
    return anatomy_analyser.analyze_video(path)

def getVideoData(path):
    print("getting video data")

    media_info = MediaInfo.parse(path)
    info = {}

    # General track
    general = next((t for t in media_info.tracks if t.track_type == "General"), None)
    if general:
        info["format"] = general.format
        # Duration in mm:ss
        if general.duration:
            time_sec = int(general.duration) / 1000
            min = int(time_sec // 60)
            sec = float(time_sec % 60)
            info["duration"] = f"{min}:{sec:.2f}"
        else:
            info["duration"] = None

        # File size in MB
        info["file_size"] = f"{int(general.file_size) / (1024*1024):.2f} MB" if general.file_size else None

        # Bit rate
        info["bit_rate"] = int(general.overall_bit_rate) if general.overall_bit_rate else None
    # Video track
    video = next((t for t in media_info.tracks if t.track_type == "Video"), None)
    if video:
        info["video_codec"] = video.format
        info["frame_rate"] = float(video.frame_rate) if video.frame_rate else None
        info["aspect_ratio"] = video.display_aspect_ratio

    return info

if __name__ == "__main__":
    # threaded=True allows Flask to handle multiple requests concurrently
    # debug=True enables auto-reload for development (optional)
    app.run(threaded=True, debug=True, host="0.0.0.0", port=5000)
