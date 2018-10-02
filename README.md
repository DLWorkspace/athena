# Athena Project
This project provides a set of Restful-APIs for AI applications.


sample:

```
img = load_img("/share1/public/isic/images/2.NV/ISIC_0034320.jpg",target_size=(224,224))
img = img_to_array(img)
imgdata = base64.encodebytes(img.tobytes())

import requests

headers = {'Content-type': 'application/json'}
payload = {'image': imgdata.decode('utf-8'),'size':(224,224,3)}
r = requests.post("http://regserver.westus2.cloudapp.azure.com/SkinLesionAnalysis", json=payload, headers=headers)
print(r.text)




tid = json.loads(r.text)["task_id"]
task_id = tid
r = requests.get("http://regserver.westus2.cloudapp.azure.com/SkinLesionAnalysis?task_id="+task_id)
print(r.text)

```
