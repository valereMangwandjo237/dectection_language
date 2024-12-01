from flask import Flask, render_template, request
import joblib
import re
import pickle

app = Flask(__name__)
app.secret_key = "176fb924321821a4dd821f5c34a696654958903137702998932c119d1388ad05"

lang = ['Arabic','Danish','Dutch','English','French','German','Greek','Hindi','Italian','Kannada','Malayalam','Portugeese','Russian','Spanish','Sweedish','Tamil','Turkish']

# upload model and vectorizer
model = pickle.load(open("models/detection_model.sav", 'rb'))
vectorizer = joblib.load('models/vectorizer.sav')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']

        text_clean = clean_function(text)
        text_clean = [text_clean]
        text_clean = vectorizer.transform(text_clean)

        y = model.predict(text_clean)[0]
        language = lang[y]

        return render_template('index.html', text=text, language=language)
    return render_template('index.html')

def clean_function(Text):
    # removing the symbols and numbers
    Text = re.sub(r'[\([{})\]!@#$,"%^*?:;~`0-9]', ' ', Text)
    
    # converting the text to lower case
    Text = Text.lower()
    Text = re.sub('http\S+\s*', ' ', Text)  # remove URLs
    Text = re.sub('RT|cc', ' ', Text)  # remove RT and cc
    Text = re.sub('#\S+', '', Text)  # remove hashtags
    Text = re.sub('@\S+', '  ', Text)  # remove mentions
    Text = re.sub('\s+', ' ', Text)  # remove extra whitespace
    
    return Text

if __name__ == '__main__':
    app.run(debug=True)