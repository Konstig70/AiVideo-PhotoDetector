#Utilizes openAI API key to create justification based on analysis before.
from openai import OpenAI
from openai.agents import create_openai_functions_agent


class VideoJustificationAgent:
    GUIDELINES = """
    1. Videos with unnatural geometry in faces or bodies are likely synthetic.
    2. Metadata inconsistencies (e.g., impossible timestamps or missings in metadata) increase suspicion.
    3. Motion anomalies (e.g., jitter, unnatural physics) are strong indicators.
    4. Give justification even if the video is likely real.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialisoi OpenAI-agentti annetulla API-avaimella."""
        self.client = OpenAI(api_key=api_key)
        self.agent = create_openai_functions_agent(
            client=self.client,
            model=model,
            description="Analyzes video data and provides synthetic video justification",
        )

    def _make_prompt(self, video_data: dict) -> str:
        """Luo promptin annetulle video datalle."""
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
        """Suorittaa analyysin annetusta videodatasta ja palauttaa perustelun."""
        prompt = self._make_prompt(video_data)
        response = self.agent.run(input=prompt)
        return response


if __name__ == "__main__":
    asked_key = input("Give your OpenAI API key: ")
    agent = VideoJustificationAgent(api_key=asked_key)

    video_data_example = {
        "geometry_anomalies": ["asymmetric eyes", "misaligned limbs"],
        "motion_anomalies": ["head jitter", "inconsistent walking pattern"],
        "metadata": {
            "creation_tool": "unknown",
            "timestamps": ["2025-01-01T12:00", "invalid"],
        },
    }

    justification = agent.analyze(video_data_example)
    print("AI detector agent justification:\n", justification)