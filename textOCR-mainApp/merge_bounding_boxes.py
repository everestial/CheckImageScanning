
# Algorithm for merging bounding boxes and texts
def merge_close_bounding_boxes(boxes_and_text):
    ## NOTE: right now, I am hard coding: h_distance = 60 and v_distance = 40
    # this can be determined dynamically in the future 
    # by taking the median distances between all pair of lines (lines of the bounding boxes)

    def recurring_merge(boxes_and_text, h_distance = 0, v_distance = 0):
        i = 0

        while i < len(boxes_and_text) - 1:
            # if wordbox_distance(boxes_and_text[i], boxes_and_text[i+1]) < dist_limit:
            # if wordbox_distance(boxes_and_text[i][0], boxes_and_text[i+1][0]) < dist_limit:
            if should_merge_word_boxes(boxes_and_text[i][0], boxes_and_text[i+1][0], h_distance = 60, v_distance = 40):
                # if there was text
                """
                [('oool\n\x0c', (1053, 51, 1157, 87)),
                ('Mia X. F\n\x0c', (65, 58, 152, 92))]
                """
                # boxes_and_text[i] = boxes_and_text[i][0] + ' ' + boxes_and_text[i+1][0]
                boxes_and_text[i][1] = boxes_and_text[i][1] + ' ' + boxes_and_text[i+1][1]

                # merge the value of the boxes
                # boxes_and_text[i][1] = merge_boxes(boxes_and_text[i][1], boxes_and_text[i+1][1])
                boxes_and_text[i][0] = merge_boxes(boxes_and_text[i][0], boxes_and_text[i+1][0])
                # boxes_and_text[i] = merge_boxes(boxes_and_text[i], boxes_and_text[i+1])

                del boxes_and_text[i + 1]
            else: i += 1
                # return True, texts_new, text2_boxes_new
        return boxes_and_text

    boxes_and_text = recurring_merge(boxes_and_text)

    ## NOTE: we may have to run it recursively
    ## TODO: improve later
    ## sort on second index and merge again
    boxes_and_text = sorted(boxes_and_text, key=lambda r:r[0][1])
    boxes_and_text = recurring_merge(boxes_and_text)

    ## sort on third index and merge again
    boxes_and_text = sorted(boxes_and_text, key=lambda r:r[0][2])
    boxes_and_text = recurring_merge(boxes_and_text)

    ## sort on third index and merge again
    boxes_and_text = sorted(boxes_and_text, key=lambda r:r[0][3])
    boxes_and_text = recurring_merge(boxes_and_text)

    ## sort on the first index again and merge again
    boxes_and_text = sorted(boxes_and_text, key=lambda r:r[0][0])
    boxes_and_text = recurring_merge(boxes_and_text)

    ## sort on the first index again and merge again
    boxes_and_text = sorted(boxes_and_text, key=lambda r:r[0][1])
    boxes_and_text = recurring_merge(boxes_and_text)

    # extracting a separate box (hoping it will be useful?)
    boxes_only = [x[0] for x in boxes_and_text]

    return boxes_and_text, boxes_only


# From two text boxes generate a larger one that covers them
# NOTE: merging co-ordinates value of two adjacent words (horizontal)
def merge_boxes(box1, box2):
    # return [min(box1[0], box2[0]), 
    #      min(box1[1], box2[1]), 
    #      max(box1[2], box2[2]),
    #      max(box1[3], box2[3])]

    """
    e.g: its is sorted by 51, 58, 77, 84
    coordinate parameters are: 
    startX, startY, endX, endY | x1, y1, x2, y2
    [
    ((1053, 51, 1157, 87), 'oool\n\x0c'),
    ((65, 58, 152, 92), 'Mia X. F\n\x0c'),
    ((544, 77, 659, 113), 'Rime\n\x0c'),
    ((148, 84, 306, 120), 'JR ADDRESS\n\x0c'),
    ((64, 84, 173, 115), '123 YOUR\n\x0c'),
    ...
    ]
    """

    # merging words; startX, startY, endX, endY | x1, y1, x2, y2
    return [min(box1[0], box2[0]), 
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])]
        #  box1[2] + box2[0] - sum(box1[0] + box1[2]) + box2[2]



# compute the distance between two words/boxes (within defined limits)
# may be replace this with the solution from here:
    # https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments 
    # https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
def wordbox_distance(word_01, word_02):

    # breakpoint()
    # try:
    #     word_01 = word_01.tolist()
    #     word_02 = word_02.tolist()
    # except TypeError:
    #     breakpoint()
    # word_01: ymin, xmin, ymax, xmax
    # word_02: ymin, xmin, ymax, xmax
    # word_01_ymin, word_01_xmin, word_01_ymax, word_01_xmax = word_01
    # word_02_ymin, word_02_xmin, word_02_ymax, word_02_xmax = word_02

    # word_01_1_left, word_01_1_bottom, word_01_1_right, word_01_1_top = word_01
    # word_02_ymin, word_02_xmin, word_02_ymax, word_02_xmax = word_02

    """
    e.g: its is sorted by 51, 58, 77, 84
    coordinate parameters are: 
    startX, startY, endX, endY | x1, y1, x2, y2
    [
    ((1053, 51, 1157, 87), 'oool\n\x0c'),
    ((65, 58, 152, 92), 'Mia X. F\n\x0c'),
    ((544, 77, 659, 113), 'Rime\n\x0c'),
    ((148, 84, 306, 120), 'JR ADDRESS\n\x0c'),
    a_x1, a_y1, a_x2, a_y2
    ((64, 84, 173, 115), '123 YOUR\n\x0c'),
    b_x1, b_y1, b_x2, b_y2

    ...
    ]
    """
    # (startX, startY, endX, endY) | x1, y1, x2, y2
    # left, top, width, height of the words (word_01 and word_02)
    a_x1, a_y1, a_x2, a_y2 = word_01
    b_x1, b_y1, b_x2, b_y2  = word_02

    h_dist = abs(b_x2 - a_x1) # 173 - 148
    v_dist = abs(b_y1 - a_y2) # 84 - 120

    # return h_dist, v_dist
    ## NOTE: in the future use both horizontal and vertical distance to make decision
    # horizontal of 50, vertical of 20?
    return max(h_dist, v_dist)


# compute the distance between two words/boxes (within defined limits)
# def check_wordbox_distance(word_01, word_02):
def should_merge_word_boxes(word_01, word_02, h_distance = 0, v_distance = 0):
    ## NOTE: in the future use both horizontal and vertical distance to make decision
    # NOTE: work on function in the future: def wordbox_distance(word_01, word_02):

    """
    e.g: its is sorted by 51, 58, 77, 84
    coordinate parameters are: 
    startX, startY, endX, endY | x1, y1, x2, y2
    [
    ((1053, 51, 1157, 87), 'oool\n\x0c'),
    ((65, 58, 152, 92), 'Mia X. F\n\x0c'),
    ((544, 77, 659, 113), 'Rime\n\x0c'),
    ((148, 84, 306, 120), 'JR ADDRESS\n\x0c'),
    a_x1, a_y1, a_x2, a_y2
    ((64, 84, 173, 115), '123 YOUR\n\x0c'),
    b_x1, b_y1, b_x2, b_y2
    ...
    ]
    """

    # def intersects(self, other):
    #     return not (self.top_right.x < other.bottom_left.x 
    #                 or self.bottom_left.x > other.top_right.x 
    #                 or self.top_right.y < other.bottom_left.y 
    #                 or self.bottom_left.y > other.top_right.y)
    
    
    # (startX, startY, endX, endY) | x1, y1, x2, y2
    # left, top, width, height of the words (word_01 and word_02)
    a_x1, a_y1, a_x2, a_y2 = word_01
    b_x1, b_y1, b_x2, b_y2  = word_02


    # h_dist = abs(b_x2 - a_x1) # 173 - 148
    # v_dist = abs(b_y1 - a_y2) # 84 - 120

    ## NOTE: this value of 60 and 40 can be determined dynamically later on 
    # if abs(b_x1 - a_x2) <= 60 and abs(b_y1 - a_y2) <= 40:
    # if abs(b_x1 - a_x2) <= 60 and abs(b_y1 - a_y2) <= 30:
    if abs(b_x1 - a_x2) <= h_distance and abs(b_y1 - a_y2) <= v_distance:
        return True
    
    # if abs(b_x2 - a_x1) <= 60 and abs(b_y1 - a_y2) <= 40:
    # if abs(b_x2 - a_x1) <= 60 and abs(b_y1 - a_y2) <= 30:
    if abs(b_x2 - a_x1) <= h_distance and abs(b_y1 - a_y2) <= v_distance:
        return True
    
    ## Checking if the two rectangles overlap
    # source: https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
    # return not (
    #     a_x2 < b_x1 or 
    #     a_x1 > b_x2 or
    #     a_y1 > b_y2 or
    #     a_y2 < b_y1)

    # adjusting for 15 and 25 unit value outside the border of the boxes 
    # NOTE: if this option works well then there is no need to use h_distance, v_distance
    # TODO: CHECK and improve as necessary.
    return not (
        a_x2 + 20 < b_x1 - 20 or 
        a_x1 - 10 > b_x2 + 10 or
        a_y1 - 10 > b_y2 + 10 or
        a_y2 + 20 < b_y1 - 20)
    


##################################################################################
################    TEST (Remove the docstring quotes and call)    ###############
"""
import cv2
from write_bounding_box import write_bounding_boxes
boxes_and_text = None 
original_image_obj = None 
file_name, file_extension = None, None 

## Step 03: Now merge the bounding boxes that are close to each other

## Step 03-A: Merge the boxes
## NOTE: The prarmeters for merge (e.g. h_distance, v_distance) can be defined in another nested function within the function called.
# merged_boxes_and_text, merged_boxes_only = merge_close_bounding_boxes(boxes_and_text, h_distance = 0, v_distance = 0)
merged_boxes_and_text, merged_boxes_only = merge_close_bounding_boxes(boxes_and_text)

print(merged_boxes_and_text)
breakpoint()

## Step 03-B (Optional): Again, write the merged boxes and the texts to the image 
# write the output image
original_image_obj_with_merged_bounding_boxes = write_bounding_boxes(original_image_obj, merged_boxes_and_text, write_text = False, rescan_text=False)
# output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)  # if writing on a gray scale
# cv2.imwrite("001_O_east03_merged_bounding_boxes.jpg", original_image_obj_with_merged_bounding_boxes)
cv2.imwrite(file_name + '_with_merged_bounding_boxes' + file_extension, original_image_obj_with_merged_bounding_boxes)

original_image_obj_with_merged_bounding_boxes_and_text = write_bounding_boxes(original_image_obj, merged_boxes_and_text, write_text = True, rescan_text=True)
# cv2.imwrite("001_O_east03_merged_bounding_boxes_with_text.jpg", original_image_obj_with_merged_bounding_boxes_and_text)
cv2.imwrite(file_name + '_with_merged_bounding_boxes_and_text' + file_extension, original_image_obj_with_merged_bounding_boxes_and_text)

print(original_image_obj_with_merged_bounding_boxes_and_text)

import json
with open('original_image_obj_with_merged_bounding_boxes.json', 'w') as json_file: 
    json.dump(original_image_obj_with_merged_bounding_boxes.tolist(), json_file, indent=4)

with open('original_image_obj_with_merged_bounding_boxes_and_text.json', 'w') as json_file: 
    json.dump(original_image_obj_with_merged_bounding_boxes_and_text.tolist(), json_file, indent=4)

print('hello')
breakpoint()
"""




