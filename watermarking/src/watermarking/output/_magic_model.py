class MagicModel:

    def __init__(self, model, output_watermarker):
        self.model = model
        self.output_watermarker = output_watermarker

    def fit(self, *args, **kwargs):
        # ✏️ à compléter

    def predict(self, *args, **kwargs):
        return self.output_watermarker.mark(self.model.predict(*args, **kwargs))
