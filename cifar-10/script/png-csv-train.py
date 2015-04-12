import sys
import argparse
import Image

"""
UASGE
-----
python png-csv-train.py dir-to-train label.csv train.csv

Input
-----
dir of train under which is the xxx.png files


Output
-----
xxx.csv file inner which each line represent
a png file, the file's header is in format:
"r-1-1,g-1-1,b-1-1,r-1-2 ... b-32-32,label"

"""

if __name__ == "__main__":
    '''
    convert png format into csv format
    '''

    # parse args
    parser = argparse.ArgumentParser()
    
    # add argument: train_dir
    parser.add_argument("train_dir", type = str,
                        help = "the train dir under which \
                        you store png files for train")
                        
    # add argment: lable_path
    parser.add_argument("label_path", type = str,
                        help = "the label file inner which \
                        each line correspond to a png file \
                        under the train_dir recording the \
                        png's label")
                        
    # add argument: train_path
    parser.add_argument("train_path", type = str,
                        help = "the path of a file to which \
                        you want to gather all png files \
                        under the train_dir, inner which \
                        each line correspnd to a png file \
                        under the train_dir recording the \
                        png's pixels[rgb] and lable")
                        
    # parse args from command
    args = parser.parse_args()

    # read in labels from label_path
    label_list = []
    with open(args.label_path, 'r') as label_file:
        # skip header 
        label_file.readline()
        # read in lines
        lines = label_file.readlines()

    # extract id:labels into dict    
    for line in lines:
        # id
        identity = int(line.strip().split(',')[0])
        # label
        label = line.strip().split(',')[1]
        # (id, label)
        label_list.append((identity, label))
    # convert label_list to dict in format:
    # {id: "label", ...}
    label_dict = dict(label_list)
    
    # write header
    header = []
    for row in range(1, 32 + 1):
        for column in range(1, 32 + 1):
            for color in ['r', 'g', 'b']:
                header.append(str(row) + '-' + str(column) + '-' + color)
    with open(args.train_path, 'w') as train_file:
        train_file.write("id" + ',' + ','.join(header) + ',' + "label\n")

    # write pixels and labels
    for image_index in range(1, 5000 + 1):
        # parse pixels
        image_pixels = []
        image_path = args.train_dir.rstrip('/') + '/' + str(image_index) + '.png'
        image = Image.open(image_path)
        for pixel_row in range(0, 32):
            for pixel_column in range(0, 32):
                pixel = image.getpixel((pixel_row, pixel_column))
                image_pixels.append(str(pixel[0]))
                image_pixels.append(str(pixel[1]))
                image_pixels.append(str(pixel[2]))
        
        # retrieval label
        image_label = label_dict[image_index]
        
        # print pixels and label
        with open(args.train_path, 'a') as train_file:
            train_file.write(str(image_index) + ',' + ','.join(image_pixels) + ',' + image_label + '\n')
    
