import sys
import json
import os

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask import request, jsonify
import base64
import yaml

import logging
from logging.config import dictConfig


from storage import Azure_Storage


import numpy as np    
import uuid

app = Flask(__name__)
api = Api(app)
verbose = True

parser = reqparse.RequestParser()

azure_storage = Azure_Storage()

class SkinLesionAnalysis(Resource):
    def post(self):
        params = request.get_json(force=True)
        #params = json.dumps(params)

        img = base64.decodebytes(bytes(params['image'],"utf-8"))
        #img = np.frombuffer(img,dtype=np.float32).tobytes()



        task_id = str(uuid.uuid4())
        task = {"task_id":task_id}
        task_msg = json.dumps(task)


        azure_storage.put_image(task_id,img)
        azure_storage.put_task(task_msg)
        

        ret = {}
        ret["task_id"] = task_id
        ret["status"] = "pending"
        ret["result"] = "Not_Available"
        resp = jsonify(ret)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["dataType"] = "json"        
        return resp

    def get(self):
        parser.add_argument('task_id')
        args = parser.parse_args()    
        task_id = args["task_id"]
        ret = {}

        results = azure_storage.get_classification_result(task_id)

        if results is not None:
            ret = {"task_id":task_id, "status":"completed", "result":results}
        else:
            ret = {"task_id":task_id, "status":"pending or wrong task_id", "result":"NA"}

        resp = jsonify(ret)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["dataType"] = "json"

        return resp        
##
## Actually setup the Api resource routing here
##
api.add_resource(SkinLesionAnalysis, '/SkinLesionAnalysis')

if __name__ == '__main__':

    app.run(debug=True,host="0.0.0.0",port="6006",threaded=True,use_reloader=False)