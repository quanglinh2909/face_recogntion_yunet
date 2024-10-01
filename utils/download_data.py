import json
import unidecode
import requests
import os
# Opening JSON file
path= 'D:\\Python\\face_recogntion_yunet\\data'
person_path= path +'\\camera-ai.person.json'
image_person_path =path + '\\camera-ai.person_image.json'

# Mở và đọc tệp JSON
with open(person_path, 'r', encoding='utf-8') as file:
    persons = json.load(file)

with open(image_person_path, 'r', encoding='utf-8') as file:
    images = json.load(file)

pathRoot = 'D:\\Python\\face_recogntion_yunet\\images'

for item in persons:
    id_person = item["_id"]['$oid']
    name = item['name']
    #bỏ tiếng việt và khoảng trắng
    name = name.replace(" ","_")
    name = unidecode.unidecode(name)
    print(name)
    pathPerson = pathRoot + '\\' + name+"@"+id_person
    for image in images:
        if image['person_id'] == id_person:
            url = image['url']
            #save image
            response = requests.get(url)
            if not os.path.exists(pathPerson):
                os.makedirs(pathPerson)
            name_image = url.split('/')[-1]
            with open(pathPerson + '\\' + name_image, 'wb') as file:
                file.write(response.content)
            print(url)



