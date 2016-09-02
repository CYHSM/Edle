"""Controls the audio output"""

from gtts import gTTS
import os
import vlc


def text_to_audio(textIn, language='en', save=False):
    # 1.) Create tts Object
    tts = gTTS(text=textIn, lang=language)

    # 2.) Save or play
    base_dir = os.path.dirname(__file__)
    mp3name = (textIn+language)

    # Make sure path name is short enough
    if len(mp3name) > 20:
        mp3name = mp3name[0:20]

    if save: #Just Save
        filename = os.path.join(base_dir, 'MP3Files/%s.mp3'%mp3name)
        tts.save(filename)
    else: #Just Play
        filename = os.path.join(base_dir, 'MP3Files/Tmp/%s.mp3'%mp3name)
        tts.save(filename)
        while not os.path.exists(filename):
            pass
        p = vlc.MediaPlayer(filename)
        p.play()
        while str(p.get_state()) != 'State.Ended':
            pass
        os.remove(filename)


def audio_to_text():
    pass
