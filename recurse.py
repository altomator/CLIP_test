# generates a list of image files contained in a folder and its subfolders
# input: root folder
# output: text file

import os
import sys
import argparse

parser = argparse.ArgumentParser(
                    prog = 'recurse.py',
                    description = 'Build a list of image files from a folder and its subfolders.'
                    )
parser.add_argument('-f', '-folderName', help='the folder to be listed',required=True)
args = parser.parse_args()
walk_dir = args.f

list_dir = "."

if not os.path.exists(walk_dir):
    print("### '%s' folder does not exist! ###\n" % walk_dir)
    quit()

##############################################
output_file = walk_dir+ ".txt"
print('walk_dir = ' + walk_dir)

print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
list_file_path = os.path.join(list_dir, output_file)

with open(list_file_path, 'wb') as list_file:
    for root, subdirs, files in os.walk(walk_dir):
        print('--\nroot = ' + root)

        for subdir in subdirs:
            print('\t- subdirectory ' + subdir)
        i=1
        for filename in [filename for filename in files if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG") or filename.endswith(".png")]:
            file_path = os.path.join(root, filename)
            if (i % 10 == 0):
                print('\t-%s : %s (full path: %s)' % (str(i), filename, file_path))
            list_file.write(('%s\n' % (file_path)).encode('utf-8'))
            i+=1
        print(i)
