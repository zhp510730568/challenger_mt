# coding = 'utf-8'
import os, time
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
start = time.clock()  # 记录处理开始时间；与最后一行一起使用，来判断输出运行时间。

root_path='../data'
data_path='../data/ai_challenger_MTEnglishtoChinese_testA_20180827_en.sgm'


def generate_test():
    counter = 0
    with open('../data/test.txt', 'w') as f:
        for line in open(data_path).readlines():
            if line.startswith('<seg id="'):
                en_sentence = line.split('">', 2)[1].split('</seg>')[0]
                print(en_sentence.split('\t', 3)[2])
                f.write('%s\n' % (en_sentence.split('\t', 3)[2]))
                counter += 1
        print(counter)


def process_decoder_output():
    with open(os.path.join(root_path, 'translation.zh'), 'r') as decoder_file:
        with open(os.path.join(root_path, 'submit.sgm'), 'w') as submit_file:
            for line in decoder_file.readlines():
                new_line = line.strip().replace(' ', '')
                submit_file.write('%s\n' % (new_line))



if __name__=='__main__':
    process_decoder_output()