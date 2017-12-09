import os
import time
from csv import reader
import codecs
from urllib.request import urlretrieve
import numpy as np
import re
from subprocess import run
from PIL import Image
import tensorflow as tf

global_directory = "FAKEPATH"

## Get categories, titles, and urls from training data
def getcategoryandurl_train():
    csvfile = open('training.csv',newline='',encoding='utf-8')
    rows = reader(csvfile,delimiter=',',quotechar='"')
    categories,urls,titles = [],[],[]

    for row in rows:
        categories.append(row[1])
        titles.append(row[3])
        urls.append(row[4])

    data = np.column_stack((titles,categories,urls))

    return data[1:]

## Get titles, and urls from unclassified data
def getcategoryandurl_classify():
    csvfile = open('classify.csv',newline='',encoding='utf-8')
    rows = reader(csvfile,delimiter=',',quotechar='"')
    urls,titles = [],[]

    for row in rows:
        titles.append(row[2])
        urls.append(row[3])

    data = np.column_stack((titles,urls))

    return data[1:]

## Download image file from given url and save to proper folder for NN training
def downloadimage_train(data):
    id=0
    ids = []
    print('Saving training images')
    for datum in data:
        base_dir = global_directory
        title = datum[0]
        category = datum[1].replace(' ','_')                ## Remove spaces for folder name
        category = re.sub(r'[\\/*?:"<>|]', "", category)    ## Remove forbidden characters for folder name
        url = datum[2]

        ## Check if training folder exists and create if not
        if not os.path.exists(os.path.join(base_dir,'product_photos_train')):
            os.mkdir(os.path.join(base_dir,'product_photos_train'))

        directory = os.path.join(base_dir,'product_photos_train',category)
        file = os.path.join(directory,str(id)+'.jpeg')

        print('\r'+str(id),end='')

        ## Check if category folder exists and create if not
        if not os.path.exists(directory):
            os.mkdir(directory)

        ## Check if file has already been downloaded and url isn't empty
        if not os.path.isfile(file) and url:
            try:
                urlretrieve(url,file)
            except:                                         ## Ignore urls that return an HTTPError
                pass

        ## Ensure image was not corrupted during download
        if os.path.isfile(file):
            try:
                im = Image.open(file)
            except:
                os.remove(file)

        ids.append(id)
        id += 1

    print('\n')
    return np.column_stack((ids,data))

## Download image file from given url and save to unclassified folder
def downloadimage_classify(data):
    id=0
    ids = []
    print('Saving unclassified images')
    for datum in data:
        base_dir = global_directory
        title = datum[0]
        url = datum[1]

        directory = os.path.join(base_dir,'product_photos_classify')
        file = os.path.join(directory,str(id)+'.jpeg')

        print('\r' + str(id), end='')

        ## Check if classification folder exists and create if not
        if not os.path.exists(directory):
            os.mkdir(directory)

        ## Check if file has already been downloaded and url isn't empty
        if not os.path.isfile(file) and url:
            try:
                urlretrieve(url,file)
            except:                                         ## Ignore urls that return an HTTPError
                pass

        ## Ensure image was not corrupted during download
        if os.path.isfile(file):
            try:
                im = Image.open(file)
            except:
                os.remove(file)

        ids.append(id)
        id += 1

    print('\n')
    return np.column_stack((ids,data))

## Retrain the final layer of an Inception-V.3 neural network for the given categories
def retrainnetwork():
    categories = os.listdir('product_photos_train')

    run(['python3', 'retrain.py', '--output_graph=retrained_graph.pb', '--output_labels=retrained_labels.txt', '--image_dir=product_photos_train'])

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def runtensor(imagefile):
    image_data = tf.gfile.FastGFile(imagefile,'rb').read()
    labels = [line.rstrip() for line in tf.gfile.GFile('retrained_labels.txt')]
    #load_graph('retrained_graph.pb')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions, = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_5 = predictions.argsort()[-5:][::-1]
        best = []
        for node_id in top_5:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
            best.append(human_string)

    return best[0]

## Classify unclassified images using retrained network
def classifyimages(data):
    base_dir = os.path.join(global_directory,'product_photos_classify')
    graph = 'retrained_graph.pb'
    labels = 'retrained_labels.txt'
    output_file = codecs.open('stackline_image_classifications.txt', 'w', 'utf-8')

    ## Load graph and labels from retrained network
    load_graph('retrained_graph.pb')
    labels = [line.rstrip() for line in tf.gfile.GFile('retrained_labels.txt')]

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for datum in data:
            id = datum[0]
            title = datum[1]

            filename = os.path.join(base_dir,str(id)+'.jpeg')

            ## Check if image failed to download
            if os.path.isfile(filename):

                ## Classify image with retrained network
                print(id)
                image_data = tf.gfile.FastGFile(filename,'rb').read()
                predictions, = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
                top_5 = predictions.argsort()[-5:][::-1]
                best = []
                for node_id in top_5:
                    human_string = labels[node_id]
                    score = predictions[node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                    best.append(human_string)

                ## Save title and classification
                output_file.write('%s\t%s\n' % (title, best[0]))

def main():
    start_time = time.time()
    print("Loading training data")
    trainingdata = getcategoryandurl_train()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Downloading training image files")
    trainingdata = downloadimage_train(trainingdata)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Retraining NN final layer")
    retrainnetwork()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Loading unclassified data")
    classifydata = getcategoryandurl_classify()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Downloading unclassified image files")
    classifydata = downloadimage_classify(classifydata)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Classifying test data")
    classifyimages(classifydata)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
