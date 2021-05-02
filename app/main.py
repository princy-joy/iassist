from flask import Flask,render_template,redirect,request
from gtts import gTTS
import os
import model_util

app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        input_image = request.files['input_image']
        image_path = "./static/{}".format(input_image.filename) 
        input_image.save(image_path)
        audio_path = "./static/{}".format(input_image.filename) + ".mp3"

        caption = model_util.caption_image(image_path)
        audio_out = gTTS(text = caption, lang = 'en',slow = False, tld='co.uk')
        audio_out.save(audio_path)

        predictions = {
            'image' : image_path,
            'caption' : caption,
            'sound' : audio_path
        }
        return render_template("index.html", predictions = predictions)
    return render_template("index.html")
	

if __name__ == '__main__':
	app.run(debug = True)
