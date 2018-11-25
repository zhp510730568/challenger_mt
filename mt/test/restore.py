#! /usr/bin/python

import os,time
import gzip

start_date = 20181026
end_date = 20181106

root_path = "../data1/"
new_path = "../data1/new_file.txt"

log_path = "./log.txt"
successed_path = "./successed.txt"

# 日志文件
log_file = open(log_path, "w+")

success_files = set()
if os.path.exists(successed_path):
    with open(successed_path, 'r') as f:
        for name in f:
            name = name.strip()
            if name:
                success_files.add(name.strip())
log_file.write("successed files: %s\n" % str(success_files))

file_names = []
for gz_file in os.listdir('./'):
    if (gz_file not in success_files) and gz_file.startswith("karos.log-") and gz_file.endswith(".gz"):
        file_date = gz_file[10:-3]
        if file_date.isdigit():
            file_date = int(file_date)
            if file_date >= start_date and start_date <= end_date:
                file_names.append(gz_file)

file_names = sorted(file_names, key=lambda name: name, reverse=False)
# 显示需要导入的归档文件
log_file.write("restore files: %s\n" % str(file_names))

count = 0
start_time = time.time()
end_time = time.time()
with open(new_path, "w") as new_file, open(successed_path, "w+") as history_file:
    for file_name in file_names:
        file_path = os.path.join(root_path, file_name)
        f = gzip.open(file_path, "r")
        for line in f:
            line = line.strip()
            new_file.write("%s\n" % (line))
            count += 1
            if count % 20000 == 0:
                end_time = time.time()
                log_file.write("time delay: %d\n" % (end_time - start_time))
                start_time = end_time
                time.sleep(1)
            if count % 100000 == 0:
                log_file.write("gz file: %s, sample log: %s\n" % (file_name, line))
        f.close()
        # 记录已写入日志文件
        history_file.write("%s\n" % file_name)
        history_file.flush()

log_file.close()