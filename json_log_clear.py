file_path = r'D:\Python项目\classification-master\tools\work_dirs\mobilenet-v3-small_8xb32_in1k\20240116_135144.log.json'

json_log = []
temp_line = 2
type_line = 4
second_line = 0
type = ''
if type_line == 4:
    type = 'MobileNet.log'
elif type_line == 3:
    type = 'MobileNet.log'


with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if temp_line == type_line:
            json_log.append(line)
            type_line += 2
        if temp_line == second_line:
            json_log.append(line)
            second_line += 2
        if temp_line == 1:
            json_log.append(line)
        temp_line += 1

write_file_path = 'D:/Python项目/classification-master/tools/work_dirs/mobilenet-v3-small_8xb32_in1k/' + type + '.json'

with open(write_file_path, 'w') as file:
    for element in json_log:
        file.write(str(element) + '\n')


print(len(json_log))
