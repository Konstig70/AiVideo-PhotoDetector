#Utilizes openAI API key to create justification based on analysis before.
from openai import OpenAI
import os
from serpapi import GoogleSearch

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
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model


    def _make_prompt(self, video_data: dict) -> str:
        """Create justification based on video_results"""
        return f"""
You are a professional synthetic video analyst.
Your job is to review the following video data and provide a human-like justification
about whether the video is synthetic or not. Focus on generalization a non tech savvy person needs to understand the justification, without needing to understand our scores in depth (ofcourse you can mention that score is a factor).
Make the justification max 6 senteces

Guidelines to consider:
{self.GUIDELINES}

Video Data:
{video_data}

Provide your answer as a concise, human-readable explanation, including which anomalies
led to your conclusion.
"""

    def analyze(self, video_data: dict) -> str:
        """Create prompt and return it"""
        prompt = self._make_prompt(video_data)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are professional AIvideo data analyst."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content

    def perform_news_cross_check(self, context):
        queries = self.generate_search_queries(context)
        results = self.search_news(queries, "")
        summary = self.analyze_news_relevance(context, results) 
        return summary

    def analyze_news_relevance(self, video_context, articles):
        """
        Prompts LLM to evaluate if articles are relevant to video content
        """
        articles_text = "\n".join([f"- {a['title']}: {a['description']}" for a in articles])
        print(articles_text)
        prompt = f"""
        You are an AI fact-checker.
        Given a video's content and a list of search queries results based on videos content, determine:
        - Are the results relevant to the video?
        - Are there inconsistencies?
        - Provide a short summary of supporting evidence.
        - You are an expert in evaluating the credibility of information. 
        - You can ignore articles that are not relevant to the video context.
        - Any none relevant articles should not be considered or even mentioned in the summary, just skip them altogether.
        - Focus on if the evidence or lack there of raises concerns for the videos authenticy or lowers them.
        - Plese write answer as if it where a scientific report. So dont use we or I in the answer. for example use words like "Infromation was found regariding x" etc.
        - Have the summary be concise and to the point. max 6 sentences.
        
        Video context:
        {video_context}
        Search query results:
        {articles_text}

        Answer in a concise summary.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        return response.choices[0].message.content


    def generate_search_queries(self, video_context):
        """
        Uses LLM to generate search queries based on extracted video content
        """
        prompt = f"""
        Generate 5 concise search queries to fact-check a video.
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
    

from io import BytesIO
from reportlab.pdfgen import canvas

class PDFGenerator:
    def __init__(self, data, geometry_results, score):
        self.data = data
        self.geometry_results = geometry_results
        self.score = score

    def generate_pdf(self):
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.drawString(50, 800, f"Suspicion score: {self.score}")
        y = 780
        c.drawString(50, y, "Data:")
        y -= 20
        for k, v in self.data.items():
            c.drawString(70, y, f"- {k.replace('_', ' ')}: {v}")
            y -= 20
        y -= 10
        c.drawString(50, y, "Human anatomy anomaly detection results:")
        y -= 20
        for k, v in self.geometry_results.items():
            c.drawString(70, y, f"- {k.replace('_', ' ')}: {v}")
            y -= 20
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
