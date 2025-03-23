import numpy as np
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import subprocess
import sys
import shutil

from flask import Flask, jsonify, request, session, Response
from flask_cors import CORS

from DcmCase import Case,RegistrationError


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="/media/jbishop/WD4/brainmets/sam_models/psam")
parser.add_argument("--uploaddir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom_upload")
parser.add_argument("--downloaddir", type=str, default="/home/jbishop/Downloads")
parser.add_argument("--niftidir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom2nifti_upload")
parser.add_argument("--datadir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/")

args = parser.parse_args()

# run in demo dir with app.app
# run in pointsam dir with what path? have to chdir instead
if True:
    os.chdir('/home/src/flaskdemo/demo')
os.environ['FLASK_APP'] = 'app.app'


# Flask Backend
app = Flask(__name__, static_folder="static")
CORS(
    app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin"
)
app.secret_key = 'test'

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route('/upload_dicom', methods=['POST'])
def upload_dicom():
    # data = request.get_json()
    file = request.files['file']
    filename = file.filename
    # filename = data.get('filename')

    if not filename:
        return jsonify({"message": "No filename received"}), 400

    session['filename'] = filename
    # save the upload
    file_path = os.path.join(args.uploaddir, filename)
    file.save(file_path)

    return jsonify({"message": f"upload complete with file: {filename}"}),200

@app.route('/preprocess', methods=['GET','POST'])
def preprocess():

    data = request.get_json()
    filename = data.get('filename', None)
    if not filename:
        return jsonify({"error": "No filename received"}), 400
    
    if False: # if using query_string and GET
        filename = request.args.get('filename')
        if not filename:
            filename = session.get('filename')

            if not filename:
                return jsonify({"error": "No filename received"}), 400
        
    if True:
        c = filename.split('.')[0]
        try:
            case = Case(c,args.uploaddir,args.niftidir,args.datadir)
        except RegistrationError:
            print('Registration failure, case {}\n\n'.format(c))

    def generate():
        result = subprocess.Popen(["python","-m","nnunet2d_predict_preprocess"],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True,
                                  bufsize=1)
        for line in iter(result.stdout.readline,""):
            print(line,end="")
            result.stdout.flush()
            yield line.strip() + '\n'
        result.stdout.close()
        result.wait()

    # return jsonify({"message": f"preprocess complete with file: {filename}"})
    return Response(generate(), mimetype="text/plain")



@app.route("/run", methods=['GET','POST'])
def run():
    filename = request.args.get('filename')
    if not filename:
        filename = session.get('filename')
        if not filename:
            return jsonify({"error": "No filename received"}), 400

    case = filename.split('.')[0]

    def generate():
        process = subprocess.Popen(
            [sys.executable, "-m", "nnunet2d_predict_wrapper"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read and yield output in real-time
        for line in iter(process.stdout.readline, ""):
            print(line, end='', flush=True)
            yield line.strip() + '\n'
            
        process.stdout.close()
        process.wait()
        
        if process.returncode != 0:
            yield f"Process exited with code {process.returncode}\n"
        else:
            yield "Process completed successfully\n"

    return Response(generate(), mimetype='text/plain')


@app.route('/postprocess', methods=['GET','POST'])
def postprocess():
    data = request.get_json()
    filename = data.get('filename', None)
    if not filename:
        return jsonify({"error": "No filename received"}), 400
    
    if False: # if using query_string and GET
        filename = request.args.get('filename')
        if not filename:
            filename = session.get('filename')
            if not filename:
                return jsonify({"error": "No filename received"}), 400
        
    c = filename.split('.')[0]
    output_zip = os.path.join(args.datadir, 'nnUNet_predictions', 'flask', f'{c}_inference.zip')
    
    # Save the output path in session
    session['output_zip'] = output_zip

    result = subprocess.Popen(["python","-m","nnunet2d_predict_postprocess"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True)
    result.wait()

    return jsonify({"message": f"postprocess complete with case: {c}"})

@app.route('/download', methods=['POST'])
def download_inference():
    # Get the output zip path from session
    output_zip = session.get('output_zip')
    if not output_zip:
        return jsonify({"error": "No output file found"}), 404

    if not os.path.exists(output_zip):
        return jsonify({"error": "Output file not found"}), 404

    # Get the filename from the path
    filename = os.path.basename(output_zip)

    # Copy the file to the download directory
    download_path = os.path.join(args.downloaddir, filename)
    shutil.copy2(output_zip, download_path)

    return jsonify({"message": f"Downloaded: {filename}"}), 200


if __name__ == "__main__":

    # something about a hot reloader when in debug mode, which double-allocates tensors on the gpu
    # can optionally force it not to use the reloader if short of gpu memory
    app.run(host=f"{args.host}", port=f"{args.port}", debug=True, use_reloader=True)
