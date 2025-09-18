#Utilizes openAI API key to create justification based on analysis before.
from openai import OpenAI


class VideoJustificationAgent:
    GUIDELINES = """
    1. Videos with unnatural geometry in faces or bodies are likely synthetic.
    2. Metadata inconsistencies (e.g., impossible timestamps or missings in metadata) increase suspicion.
    3. Motion anomalies (e.g., jitter, unnatural physics) are strong indicators.
    4. Give justification even if the video is likely real.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model


    def _make_prompt(self, video_data: dict) -> str:
        """Create justification based on video_results"""
        return f"""
You are a professional synthetic video analyst.
Your job is to review the following video data and provide a human-like justification
about whether the video is synthetic or not.

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
    justification = agent.analyze(video_data)
    print("AI detector agent justification:\n", justification)

if __name__ == "__main__":
    main()
