from flask import Flask, render_template, request

app = Flask(__name__)

switch_flag = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global switch_flag
    if request.method == 'POST':
        if 'translate_button' in request.form:
            user_input = request.form['text_input']
            return render_template('index.html', input_text=user_input)
        
        elif 'switch_button' in request.form:
            switch_flag = not switch_flag
            switch_button = switch_flag
            return render_template('index.html', input_text=None, switch_button=switch_button)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port="5005",debug=True)
