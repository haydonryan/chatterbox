try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


from .vc import ChatterboxVC
# Reenaable if decide to make multi-lingual
#from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
