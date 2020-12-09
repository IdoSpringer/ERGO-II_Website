from flask import Flask, render_template, send_file, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
from Predict import predict
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = b"\x82\x95\xef\x02\x1c\x08bz'\xc40\x1a\xed4\xdf\xe0"
app.config['UPLOAD_FOLDER'] = "upload"
app.config['baseURI'] = "http://ergoii.cs.biu.ac.il"


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    print('home', file=sys.stdout)
    error_message = False
    if request.method == 'POST':
        print('home post', file=sys.stdout)
        try:
            # check if the post request has the file part
            if len(request.files) == 0:
                flash('No file part')
                return redirect(request.url)
            # get first file
            file = request.files.values().__next__()
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                print('file uploaded', file=sys.stdout)
                filename = secure_filename(file.filename)
                test_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(test_file_path, file=sys.stdout)
                file.save(test_file_path)
                # check for number of lines
                with open(test_file_path, "r") as f:
                    lines = len(f.readlines())
                max_len = 50000
                if lines > max_len:
                    raise IndexError('file is too long')
                # read form
                tcr_encoding_model = request.form['model_type']
                dataset = request.form['dataset']
                use_alpha = 'use_alpha' in request.form
                use_vj = 'use_vj' in request.form
                use_mhc = 'use_mhc' in request.form
                use_t_type = 'use_t_type' in request.form
                # version flags
                version = ''
                version += '1'
                if dataset == 'vdjdb':
                    version += 'v'
                elif dataset == 'mcpas':
                    version += 'm'
                if tcr_encoding_model == 'AE':
                    version += 'e'
                elif tcr_encoding_model == 'LSTM':
                    version += 'l'
                if use_alpha:
                    version += 'a'
                if use_vj:
                    version += 'j'
                if use_mhc:
                    version += 'h'
                if use_t_type:
                    version += 't'
                print('version: ' + version, file=sys.stdout)
                df = predict(version, test_file_path)
                df.to_csv(app.config['UPLOAD_FOLDER'] + '/results.csv', index=False)
                os.remove(test_file_path)
                return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename='results.csv')
        except IndexError:
            error_message = True
            print('long file error', file=sys.stderr)
            os.remove(test_file_path)
            return render_template("too_long_input_file.html", error_message=error_message)
        except:
            error_message = True
            os.remove(test_file_path)
    return render_template("home.html", error_message=error_message)


@app.route("/help")
def help():
    return render_template("help.html")


@app.route("/example")
def example():
    return render_template("example.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/download_example")
def download_example():
    return send_from_directory(directory="static", filename="example.csv")


@app.route("/download_result")
def download_result():
    return send_from_directory(directory="static", filename="results_1meaj.csv")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8086))
    print('start app', file=sys.stdout)
    app.run(host='0.0.0.0', port=port, debug=True)
