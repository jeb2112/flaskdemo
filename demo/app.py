import numpy as np
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import subprocess

from flask import Flask, jsonify, request, session, Response
from flask_cors import CORS

from DcmCase import Case,RegistrationError


# ---------------------------------------------------------------------------- #
# run model
# ---------------------------------------------------------------------------- #

# radnec segmenation by nnUNet
# def segment(self,dpath=None):
#     print('segment tumour')
#     if dpath is None:
#         dpath = os.path.join(self.localstudydir,'nnunet')
#         if not os.path.exists(dpath):
#             os.mkdir(dpath)
#     for dt,suffix in zip(['t1+','flair'],['0000','0003']):
#         if os.name == 'posix':
#             l1str = 'ln -s ' + os.path.join(self.localstudydir,dt+'_processed.nii.gz') + ' '
#             l1str += os.path.join(dpath,self.studytimeattrs['StudyDate']+'_'+suffix+'.nii.gz')
#         elif os.name == 'nt':
#             l1str = 'copy  \"' + os.path.join(self.localstudydir,dt+'_processed.nii.gz') + '\" \"'
#             l1str += os.path.join(dpath,os.path.join(dpath,self.studytimeattrs['StudyDate']+'_'+suffix+'.nii.gz')) + '\"'
#         os.system(l1str)

#     command = 'conda run -n ptorch nnUNetv2_predict '
#     command += ' -i ' + dpath
#     command += ' -o ' + dpath
#     command += ' -d137 -c 3d_fullres'
#     res = os.system(command)
            
#     sfile = self.studytimeattrs['StudyDate'] + '.nii.gz'
#     segmentation,affine = self.loadnifti(sfile,dpath)
#     ET = np.zeros_like(segmentation)
#     ET[segmentation == 3] = 1
#     WT = np.zeros_like(segmentation)
#     WT[segmentation > 0] = 1
#     self.writenifti(ET,os.path.join(self.localstudydir,'ET.nii'),affine=affine)
#     self.writenifti(WT,os.path.join(self.localstudydir,'WT.nii'),affine=affine)
#     if False:
#         os.remove(os.path.join(dpath,sfile))

#     return 

# main

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="/media/jbishop/WD4/brainmets/sam_models/psam")
parser.add_argument("--uploaddir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom_upload")
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

# @app.route("/static/<path:path>")
# def static_server(path):
    # return app.send_static_file(path)

# @app.route('/get_filename')
# def get_filename():
#     filename = session.get('filename')  # Retrieve filename from session
#     if filename:
#         return jsonify({"filename": filename})
#     return jsonify({"message": "No file uploaded yet"}), 404


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

@app.route("/run", methods=['GET','POST'])
def run():

    filename = request.args.get('filename')
    if not filename:
        filename = session.get('filename')

        if not filename:
            return jsonify({"error": "No filename received"}), 400

    case = filename.split('.')[0]

    def generate_stdout():
        result = subprocess.Popen(["python","-m","nnunet2d_predict_wrapper"],
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

    # result = subprocess.run(["python","-m","nnUNetv2_predict_wrapper"],shell=False,capture_output=False,text=True)

    # return jsonify({"message": f"inference complete with case: {case}"})
    return Response(generate_stdout(), mimetype="text/plain")


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

    result = subprocess.Popen(["python","-m","nnunet2d_predict_postprocess"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True)
    result.wait()

    return jsonify({"message": f"postprocess complete with case: {c}"})
    # return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":

    # something about a hot reloader when in debug mode, which double-allocates tensors on the gpu
    # can optionally force it not to use the reloader if short of gpu memory
    app.run(host=f"{args.host}", port=f"{args.port}", debug=True, use_reloader=True)
