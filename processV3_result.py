from os import listdir
from os.path import join
import re
import json

root = '/data/csyData/dataset'

path = join(root, 'api_result', 'clean_md')
inputfile_path = join(root, 'clean', 'clean_md')
dirlist = listdir(path)


output = []
for i in dirlist:
    dirpath = join(path,i)
    filelist = listdir(dirpath)
    for file in filelist:
        med_recordpath = join(inputfile_path,i,file)
        filepath = join(dirpath,file)
        with open(filepath) as f:
            filedata = f.readlines()

        V3result = {}
        key = False
        pattern = r'\*\*(.*?)\*\*'
        for text in filedata:
            if bool(re.search(r'\d+\.\s', text)) and bool(re.search(r'\*\*', text)):
                matches = re.findall(pattern, text, re.DOTALL)
                if len(matches) >= 2:
                    cleaned_matches = []
                    for match in matches:
                        cleaned_match = re.sub(r'^\d+(?:-\d+)?\.\s*', '', match.strip())
                        cleaned_matches.append(cleaned_match)
                    key = "&".join(cleaned_matches)
                else:
                    key = re.sub(r'^\d+(?:-\d+)?\.\s*', '', matches[0].strip())
                    key = key.strip('[]')

                V3result[key] = ""
            elif key:
                V3result[key] += text.replace("---\n\n","")
        
        with open(med_recordpath) as f:
            instruction = f.read().split('# 初步诊断：')[0]
        output.append({
                    'instruction': instruction,
                    "input": "",
                    'output': V3result,
                    'system': ""
                })

with open(join(root, 'medical_clean1.json'),'w') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
