from flask import Flask, redirect, url_for, request, render_template, Response, redirect
from gtts import gTTS
#from gevent.pywsgi import WSGIServer
import os
# import model_util
import lstm_model_util as model_util

app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        input_image = request.files['input_image']
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))

        image_path = "./static/{}".format(input_image.filename)
        image_path_save = "{}/static/{}".format(APP_ROOT, input_image.filename) 
 
        input_image.save(image_path_save)
        audio_path = "./static/{}".format(input_image.filename) + ".mp3"        
        audio_path_save = "{}/static/{}".format(APP_ROOT, input_image.filename) + ".mp3"
        caption = model_util.caption_image(image_path_save)
        audio_out = gTTS(text = caption, lang = 'en',slow = False, tld='co.uk')
        audio_out.save(audio_path_save)

        predictions = {
            'image' : image_path,
            'caption' : caption,
            'sound' : audio_path
        }
        return render_template("index.html", predictions = predictions)
    return render_template("index.html")
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
