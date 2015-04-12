import sys
import argparse
import Image

"""
UASGE
-----
python png-csv-test.py dir-to-test test.csv

Input
-----
dir of test under which is the xxx.png files


Output
-----
xxx.csv file inner which each line represent
a png file, the file's header is in format:
"r-1-1,g-1-1,b-1-1,r-1-2 ... b-32-32"

"""

if __name__ == "__main__":
    '''
    convert png format into csv format
    '''

    # parse args
    parser = argparse.ArgumentParser()

    # add argument: test_dir
    parser.add_argument("test_dir", type = str,
                        help = "the test dir under which \
                        you store png files for test")
                        
    # add argument: test_path
    parser.add_argument("test_path", type = str,
                        help = "the path of a file to which \
                        you want to gather all png files \
                        under the test_dir, inner which \
                        each line correspnd to a png file \
                        under the test_dir recording the \
                        png's pixels[rgb]")

    # parse args from command
    args = parser.parse_args()

    # write header
    header = []
    for row in range(1, 32 + 1):
        for column in range(1, 32 + 1):
            for color in ['r', 'g', 'b']:
                header.append(str(row) + '-' + str(column) + '-' + color)
    with open(args.test_path, 'w') as test_file:
        test_file.write("id" + ',' + ','.join(header) + ',' + "label\n")

    # write pixels and labels
    for image_index in range(1, 300000 + 1):
        # parse pixels
        image_pixels = []
        image_path = args.test_dir.rstrip('/') + '/' + str(image_index) + '.png'
        image = Image.open(image_path)
        for pixel_row in range(0, 32):
            for pixel_column in range(0, 32):
                pixel = image.getpixel((pixel_row, pixel_column))
                image_pixels.append(str(pixel[0]))
                image_pixels.append(str(pixel[1]))
                image_pixels.append(str(pixel[2]))
        
        # print pixels and label
        with open(args.test_path, 'a') as test_file:
            test_file.write(str(image_index) + ',' + ','.join(image_pixels) + '\n')
        
    
        
    
