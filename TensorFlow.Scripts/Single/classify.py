import tensorflow as tf, sys
import os.path

def getCurrentFolder():
    return os.path.dirname(os.path.abspath(__file__))

def getBaseFolder():
    return os.path.join(getCurrentFolder(), os.path.abspath('../../Sets/Single'))

if __name__ == '__main__':

    image_path =  '{0}\sample_flower.jpg'.format(getBaseFolder())
    graph_path = '{0}\output_graph.pb'.format(getBaseFolder())
    labels_path = '{0}\output_labels.txt'.format(getBaseFolder())
    
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # https://opensource.com/article/17/12/tensorflow-image-classification-part-1

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
        in tf.gfile.GFile(labels_path)]
    
    # Unpersists graph from file
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    # Feed the image_data as input to the graph and get first prediction
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, 
        {'DecodeJpeg/contents:0': image_data})
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))