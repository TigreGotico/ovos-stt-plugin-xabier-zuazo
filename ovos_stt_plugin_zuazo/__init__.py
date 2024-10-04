from ovos_stt_plugin_whisper import WhisperSTT
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG


class XabierZuazoSTT(STT):
    MODELS = ["zuazo/whisper-tiny-pt",
              "zuazo/whisper-tiny-pt-old",
              "zuazo/whisper-tiny-gl",
              "zuazo/whisper-tiny-eu",
              "zuazo/whisper-tiny-eu-from-es",
              "zuazo/whisper-tiny-eu-cv16_1",
              "zuazo/whisper-tiny-es",
              "zuazo/whisper-tiny-ca",
              "zuazo/whisper-small-pt",
              "zuazo/whisper-small-pt-old",
              "zuazo/whisper-small-gl",
              "zuazo/whisper-small-eu",
              "zuazo/whisper-small-eu-from-es",
              "zuazo/whisper-small-eu-cv16_1",
              "zuazo/whisper-small-es",
              "zuazo/whisper-small-ca",
              "zuazo/whisper-base-pt",
              "zuazo/whisper-base-pt-old",
              "zuazo/whisper-base-gl",
              "zuazo/whisper-base-eu",
              "zuazo/whisper-base-eu-from-es",
              "zuazo/whisper-base-eu-cv16_1",
              "zuazo/whisper-base-es",
              "zuazo/whisper-base-ca",
              "zuazo/whisper-medium-pt",
              "zuazo/whisper-medium-pt-old",
              "zuazo/whisper-medium-gl",
              "zuazo/whisper-medium-eu",
              "zuazo/whisper-medium-eu-from-es",
              "zuazo/whisper-medium-eu-cv16_1",
              "zuazo/whisper-medium-es",
              "zuazo/whisper-medium-ca",
              "zuazo/whisper-large-pt",
              "zuazo/whisper-large-pt-old",
              "zuazo/whisper-large-gl",
              "zuazo/whisper-large-eu",
              "zuazo/whisper-large-eu-from-es",
              "zuazo/whisper-large-eu-cv16_1",
              "zuazo/whisper-large-es",
              "zuazo/whisper-large-ca",
              "zuazo/whisper-large-v2-pt",
              "zuazo/whisper-large-v2-gl",
              "zuazo/whisper-large-v2-eu",
              "zuazo/whisper-large-v2-eu-from-es",
              "zuazo/whisper-large-v2-eu-cv16_1",
              "zuazo/whisper-large-v2-es",
              "zuazo/whisper-large-v2-ca",
              "zuazo/whisper-large-v2-pt-old",
              "zuazo/whisper-large-v3-pt",
              "zuazo/whisper-large-v3-gl",
              "zuazo/whisper-large-v3-eu",
              "zuazo/whisper-large-v3-eu-from-es",
              "zuazo/whisper-large-v3-eu-cv16_1",
              "zuazo/whisper-large-v3-es",
              "zuazo/whisper-large-v3-ca",
              "zuazo/whisper-large-v3-pt-old"]

    def __init__(self, config=None):
        super().__init__(config)
        model_id = self.config.get("model")
        l = self.lang.split("-")[0]
        if not model_id and l in ["pt", "es", "ca", "eu", "gl"]:
            model_id = f"zuazo/whisper-small-{l}"
        if not model_id:
            raise ValueError("invalid model")
        if model_id == "small":
            model_id = f"zuazo/whisper-small-{l}"
        elif model_id == "medium":
            model_id = f"zuazo/whisper-medium-{l}"
        elif model_id == "large" or model_id == "large-v3":
            model_id = f"zuazo/whisper-large-v3-{l}"
        elif model_id == "large-v2":
            model_id = f"zuazo/whisper-large-v2-{l}"
        elif model_id == "large-v1":
            model_id = f"zuazo/whisper-large-{l}"
        self.config["model"] = model_id
        self.config["lang"] = l
        self.config["ignore_warnings"] = True
        valid_model = model_id in self.MODELS
        if not valid_model:
            LOG.info(f"{model_id} is not default model_id ({self.MODELS}), "
                     f"assuming huggingface repo_id or path to local model")
        self.stt = WhisperSTT(self.config)

    def execute(self, audio, language=None):
        return self.stt.execute(audio, language)

    @property
    def available_languages(self) -> set:
        return {"pt", "es", "ca", "eu", "gl"}
