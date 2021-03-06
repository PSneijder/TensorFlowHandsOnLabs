import tensorflow as tf
import sys
import os


def getCurrentFolder():
    return os.path.dirname(os.path.abspath(__file__))

def getBaseFolder():
    return os.path.join(getCurrentFolder(), os.path.abspath('../../Sets/Multiple'))


# change this as you see fit
image_path = "{0}/sample_flower.jpg".format(getBaseFolder())
label_path = "{0}/labels.txt".format(getBaseFolder())
model_path = "{0}/output_graph.pb".format(getBaseFolder())

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(label_path)]

# Unpersists graph from file
with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    

    filename = "results.txt"    
    with open(filename, 'a+') as f:
        f.write('\n**%s**\n' % (image_path))
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            f.write('%s (score = %.5f)\n' % (human_string, score))
