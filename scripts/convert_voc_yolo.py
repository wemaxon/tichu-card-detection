import xml.etree.ElementTree as ET
import os
import sys
from glob import glob

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_fn, labels_dir):
    with open(xml_fn) as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
    
    # Build the output file path in the specified labels directory
    base_name = os.path.basename(xml_fn).replace(".xml", ".txt")
    txt_fn = os.path.join(labels_dir, base_name)
    
    with open(txt_fn, 'w') as out_file:
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {bb[0]:0.6f} {bb[1]:0.6f} {bb[2]:0.6f} {bb[3]:0.6f}\n")

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} images_dir labels_dir classes.names")
    print(
        f"Ex: {sys.argv[0]} "
        "data/synthetic/color/train/labels "
        "data/synthetic/color/train/labels "
        "configs/classes.names"
    )
    print("From XML files in images_dir, convert them to txt files with annotation information.")
    sys.exit(1)

images_dir = sys.argv[1]
labels_dir = sys.argv[2]
classes_fn = sys.argv[3]

if not os.path.isdir(images_dir):
    print(f"{images_dir} is not a directory")
    sys.exit(1)
if not os.path.isfile(classes_fn):
    print(f"Classes file {classes_fn} is not a file")
    sys.exit(1)

with open(classes_fn, "r") as f:
    classes = f.read().splitlines()
classes = [c for c in classes if c != '']
print("Classes:", classes, "Total:", len(classes))

# Create the labels folder if it doesn't exist
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

for i, xml_fn in enumerate(glob(os.path.join(images_dir, "*.xml"))):
    convert_annotation(xml_fn, labels_dir)
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} files", flush=True)
