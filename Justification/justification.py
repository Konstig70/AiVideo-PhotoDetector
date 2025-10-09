#Utilizes openAI API key to create justification based on analysis before.
from openai import OpenAI
import os
import requests
from serpapi import GoogleSearch
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader



class VideoJustificationAgent:
    GUIDELINES = """
    1. Videos with unnatural geometry in faces or bodies are likely synthetic.
    2. Metadata inconsistencies (e.g., impossible timestamps or missings in metadata) increase suspicion.
    3. Motion anomalies (e.g., jitter, unnatural physics) are strong indicators.
    4. Give justification even if the video is likely real.
    5. The anatomy_anomaly_rating score corresponds with these values Anomaly score < 0.025: Likely a real video; 0.025 ≤ Anomaly score < 0.050: Probably a real video but some minor anomalies were detected; 0.05 ≤ Anomaly score < 0.075: Most possibly a low quality or highly edited video with some synthetic tampering, some anomalies were detected; 0.075 ≤ Anomaly score ≤ 0.1: Probably synthetic video, quite many anomalies; Anomaly score > 0.1: Highly suspicious, most likely a synthetic video, many anomalies detected. 
    6. The anatomy_anomaly_rating score should not be the only factor to consider, conscider also the amount and type of anomalies spotted.
    7. Be confident if the scores say likely real video you should mainly point it to be a real video (ofcourse mention that some anomalies were found but the overall analysis needs to match the results)
    8. Even though you understand the anomaly rating keep the explanation simple so that a non tech savvy person can understand it i.e dont mention the actual score, mentioning that some amount of frames contained anomalies is ok just dont mention the anomaly score itself just reference it.
    9. Motion score indicates how much motion is in the scene, higher scores indicate more motion, which in turn raises the amount of anomalies even in real videos. So always take motion score into account when justifying, i.e with enough movement even a real video can have quite many anomalies.  
    10. Make your own decisions based on the data, dont just parrot back the data, but ofcourse dont make assumptions that arent present.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.apiKey = api_key
        self.model = model
        self.messages = []


    def _make_prompt(self, video_data: dict) -> str:
        frame_files = {
            "finger_anomaly_frame": "../finger_anomaly_frame_.png",
            "limb_anomaly_frame": "../limb_anomaly_frame_.png",
            "face_anomaly_frame": "../face_anomaly_frame_.png",
            "no_anomaly_frame": "../no_anomaly_frame_.png"
        }

        ready_frames = {}

        for key, relative_path in frame_files.items():
            full_path = os.path.join(os.getcwd(), relative_path)
            if os.path.isfile(full_path):
                with open(full_path, 'rb') as f:
                    ready_frames[key] = io.BytesIO(f.read())
            else:
                print(f"Warning: {full_path} not found, skipping {key}")
        """Create justification based on video_results"""
        return f"""
You are a professional synthetic video analyst.
Your job is to review the following video data and 4 frames from the video (three of them contain an anomaly one doesnt). After reviewing provide a human-like justification
about whether the video is synthetic or not with a max 6-word summary in the start with a score related emoji (for example a real video start could be "The video is real :Thumbsup:"). Focus on generalization a non tech savvy person needs to understand the justification, without needing to understand our scores in depth (ofcourse you can mention that score is a factor).
Make the justification max 6 senteces. IF one of the frames is not loading, just ignore it and continue with the rest, the absense of frames shouldnt stop you from giving a justification.
Dont mention the actual scores in the justification, just reference them in a human understandable way, also dont reference how many frames you got and how many were anomalous the frames are there so that you get a feel of how a certain anomaly or no anomaly might look (ofcourse you can mention that you analyzed frames and found that the anomalies were present).

Guidelines to consider:
{self.GUIDELINES}

Video Data and frames:
{video_data}
{ready_frames}
Provide your answer as a concise, human-readable explanation, including which anomalies
led to your conclusion. 
"""

    def analyze(self, video_data: dict) -> str:
        prompt = self._make_prompt(video_data)

        # Append user prompt to conversation memory
        self.messages.append({"role": "user", "content": prompt})

        # Convert conversation into Gemini 'contents' format
        contents = []
        for msg in self.messages:
            contents.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        payload = {
            "contents": contents
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Extract assistant reply
        reply = result["candidates"][0]["content"][0]["text"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply
        

    def perform_news_cross_check(self, context, api_key):
        queries = self.generate_search_queries(context)
        results = self.search_news(queries, api_key)
        summary = self.analyze_news_relevance(context, results) 
        return summary
    
    def search_news(self, queries, key):
        """
        Perform a Google News search using SerpApi and return organic_results.

        Args:
            queries (str): The search query.

        Returns:
            list: List of organic results from the search.
        """
        # Make sure you have your SERPAPI_API_KEY set in your environment variables
        params = {
            "engine": "google",
            "q": queries,
            "api_key": key
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Return organic results (main news articles)
        return results.get("organic_results", [])

    def analyze_news_relevance(self, video_context, articles):
        """
        Prompts LLM to evaluate if articles are relevant to video content, keeping conversation memory.
        """
        # Initialize messages if not already done
        if not hasattr(self, "messages"):
            self.messages = [{"role": "system", "content": "You are an AI fact-checker and expert in evaluating video content credibility."}]

        # Format articles into text

        # Build prompt
        prompt = f"""
        This is a continuation of the analysis you just performed now your job is to evaluate if the following articles are relevant to the video content and if so do they boost its chance of being real or synthetic.
        - Are the results relevant to the video?
        - Are there inconsistencies?
        - Provide a short summary of supporting evidence.
        - Ignore irrelevant articles entirely.
        - Focus on whether evidence or lack thereof raises concerns for the video's authenticity.
        - Write the answer as a scientific report (no "we" or "I"). Max 6 sentences.
        - If no relevant articles are found, state that the news crosscheck did not provide anything that found help either case.


        Video context:
        {video_context}
        Search query results:
        {articles}
        
        Answer in a concise summary.
        """

        # Append user prompt to messages
        self.messages.append({"role": "user", "content": prompt})

        reply = ""
        
        self.messages.append({"role": "assistant", "content": reply})

        return reply


    def generate_search_queries(self, video_context):
        """
        Uses LLM to generate search queries based on extracted video content
        """
        prompt = f"""
        Generate 5 concise search queries to fact-check a video. If the context mentions or is in another language you should provide some queries in the mentioned/used language. For example if the context mentions finland or finnish people have atleast two queries in finnish
        Video context:
        {video_context}
        Only return queries as a list of strings dont number them.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        queries = response.choices[0].message.content

        # Parse into Python list
        import ast
        queries = ast.literal_eval(queries) if queries.startswith('[') else queries.split("\n")
        print("Generated queries:", queries)
        return queries
    

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from io import BytesIO
import os
from reportlab.lib.utils import ImageReader

class PDFGenerator:
    def __init__(self, data, geometry_results, score, response, beginning):
        self.data = data
        self.geometry_results = geometry_results
        self.score = score
        self.response = response
        self.beginning = beginning
        self.image_paths = {
            "finger_anomaly_frame": "../finger_anomaly_frame_.png",
            "limb_anomaly_frame": "../limb_anomaly_frame_.png",
            "face_anomaly_frame": "../face_anomaly_frame_.png",
            "no_anomaly_frame": "../no_anomaly_frame_.png"
        }
        # Optional descriptions for each image
        self.image_descriptions = {
            "finger_anomaly_frame": "Detected finger anomaly highlighted in red.",
            "limb_anomaly_frame": "Detected limb anomaly highlighted in red.",
            "face_anomaly_frame": "Detected face anomaly highlighted in red.",
            "no_anomaly_frame": "No anomalies detected in this scan."
        }

    def _draw_wrapped_text(self, c, text, x, y, max_width, font_name="Helvetica", font_size=12, leading=14):
        lines = simpleSplit(text, font_name, font_size, max_width)
        for line in lines:
            if y < 50:  # bottom margin
                c.showPage()
                y = A4[1] - 50
                c.setFont(font_name, font_size)
            c.drawString(x, y, line)
            y -= leading
        return y

    def generate_pdf(self):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Add score
        c.setFont("Helvetica-Bold", 14)
        y = height - 50
        y = self._draw_wrapped_text(c, f"Verdict: {self.beginning}", 50, y, max_width=width-100, font_name="Helvetica-Bold", font_size=14, leading=18)

        # Add data
        y -= 10
        c.setFont("Helvetica", 12)
        y = self._draw_wrapped_text(c, "Data:", 50, y, max_width=width-100, font_size=12)
        for k, v in self.data.items():
            y = self._draw_wrapped_text(c, f"- {k.replace('_', ' ')}: {v}", 70, y, max_width=width-100, font_size=12)

        # Add geometry results
        y -= 10
        y = self._draw_wrapped_text(c, "Human anatomy anomaly detection results:", 50, y, max_width=width-100, font_size=12)
        for k, v in self.geometry_results.items():
            y = self._draw_wrapped_text(c, f"- {k.replace('_', ' ')}: {v}", 70, y, max_width=width-100, font_size=12)

        # Add response
        y -= 10
        y = self._draw_wrapped_text(c, "Analysis:", 50, y, max_width=width-100, font_size=12)
        y = self._draw_wrapped_text(c, self.response, 70, y, max_width=width-100, font_size=12)

        y -= 10
        y = self._draw_wrapped_text(c, "Checked by: S&K-software analyzer", 50, y, max_width=width-100, font_size=12)
        # Add images in 2x2 grid safely with descriptions
        img_width = 250
        img_height = 180
        margin_x = 50
        spacing_x = 20
        spacing_y = 40  # extra space for caption
        images_per_row = 2
        x_start = margin_x
        y_start = y - img_height

        idx = 0
        for img_name, img_path in self.image_paths.items():
            row = idx // images_per_row
            col = idx % images_per_row
            x_pos = x_start + col * (img_width + spacing_x)
            y_pos = y_start - row * (img_height + spacing_y)

            # Start new page if image would overflow
            if y_pos < 100:
                c.showPage()
                y_pos = height - img_height - 50

            # Draw image
            if os.path.exists(img_path):
                try:
                    img = ImageReader(img_path)
                    c.drawImage(img, x_pos, y_pos, width=img_width, height=img_height)
                except Exception:
                    c.drawString(x_pos, y_pos + img_height / 2, f"Failed to load {img_name}")
            else:
                c.drawString(x_pos, y_pos + img_height / 2, f"Image {img_name} not found")

            # Draw description under image
            description = self.image_descriptions.get(img_name, "")
            c.setFont("Helvetica-Oblique", 10)
            self._draw_wrapped_text(c, description, x_pos, y_pos - 15, max_width=img_width, font_size=10, leading=12)

            idx += 1

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer




def main():

    video_data = {
"""
device info: https://clipchamp.com/
timestamps: 2025-09-18 12:38:13.731 UTC
resolution: 1920x1080
codec: avc1
Suspicion score: 0

Geometrical and human anatomy detection results:

total frames: 223
anomaly frames: 211
anomaly rating: 0.286 which equates to : Highly suspicious most likely a synthethic video, many anomalies detected
finger anomaly frames: 104
arm length ratio anomaly frames : 199
shoulder to shoulder width anomaly frames: 7
face distance anomaly frames: 44
motion score: 1.0949195623397827
"""
}

    asked_key = input("Tell your API KEY: ")
    agent = VideoJustificationAgent(api_key=asked_key)
    print(agent.search_news(["Foo Fighters recent interviews Dave Grohl"], ""))
    

if __name__ == "__main__":
    main()
