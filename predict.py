# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import multiskin

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = multiskin.model.Model(hf_token="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")

    def predict(
        self,
        prompt: str = Input(description="Prompt to infer."),
        nif: int = Input(description="Number of inference steps."),
        width: int = Input(description="Width of generated image."),
        height: int = Input(description="Height of generated image."),
    ) -> Path:
        """Run a single prediction on the model"""
        infer_config = multiskin.model.InferConfig(prompts=[prompt], num_inference_steps=nif, width=width, height=height)
        try:
            generated_filenames = self.model.infer(infer_config=infer_config)
            return Path(generated_filenames[0])
        except:
            return {
                "error": 1,
                "message": "Failed."
            }
