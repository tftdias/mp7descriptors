__author__ = 'Tiago'


import cv2
import numpy as np
import sys


if __name__ == "__main__":

    # obtain image filename
    if len(sys.argv) < 2:
        print 'Please, specify image filename.'
        sys.exit(0)
    image_filename = sys.argv[1]

    # obtain destination description filename
    if len(sys.argv) < 3:
        print 'Please, specify destination descriptor filename.'
        sys.exit(0)
    dest_filename = sys.argv[2]

    image = cv2.imread(image_filename)

    hists = np.zeros([16, 5], dtype=int)

    direction_filters = np.array([[1, -1, 1, -1],                       #vertical
                                  [1, 1, -1, -1],                       #horizontal
                                  [np.sqrt(2), 0, 0, -np.sqrt(2)],      #45 diagonal
                                  [0, np.sqrt(2), -np.sqrt(2), 0],      #135 diagonal
                                  [2, -2, -2, 2]                        #nao direccional
                        ]).T
    direction_threshold = 50
    
    common_subimage_height = image.shape[0]/4
    common_subimage_width = image.shape[1]/4

    subimage_index = 0

    for i in np.arange(0, image.shape[0], common_subimage_height):
        subimage_height = common_subimage_height if (image.shape[0]-i >= 2*common_subimage_height) else image.shape[0]-i
        for j in np.arange(0, image.shape[1], common_subimage_width):
            subimage_width = common_subimage_width if (image.shape[1]-j >= 2*common_subimage_width) else image.shape[1]-j
            subimage = image[i:i+subimage_height, j:j+subimage_width]
            
            #DEBUG
            # cv2.namedWindow('SubImage')
            # print 'Subimage', subimage_index, 'from image - [', i, ':', i+subimage_height, ',', j, ':', j+subimage_width, ']'
            # cv2.imshow('SubImage', subimage)
            # cv2.waitKey(0)

            common_block_height = subimage.shape[0]/33
            common_block_width = subimage.shape[1]/33

            ii_begin = 0

            for ii_end in np.arange(common_block_height, subimage.shape[0], common_block_height):
                if subimage.shape[0]-ii_end >= common_block_height:
                    subimage_block_line = subimage[ii_begin:ii_end, :]
                else:
                    subimage_block_line = subimage[ii_begin:subimage.shape[0], :]

                jj_begin = 0
                for jj_end in np.arange(common_block_width, subimage.shape[1], common_block_width):
                    if subimage.shape[1]-jj_end >= common_block_width:
                        block = subimage_block_line[:, jj_begin:jj_end]
                    else:
                        block = subimage_block_line[:, jj_begin:subimage.shape[1]]

                    #DEBUG
                    # cv2.namedWindow('Block')
                    # print '\tBlock', ii_begin/common_block_height*33+jj_begin/common_block_width, '- size', block.shape[0], 'x', block.shape[1]
                    # cv2.imshow('Block', block)

                    subblock1_mean = np.mean(block[0:block.shape[0]/2, 0:block.shape[1]/2])
                    subblock2_mean = np.mean(block[0:block.shape[0]/2, block.shape[1]/2:block.shape[1]])
                    subblock3_mean = np.mean(block[block.shape[0]/2:block.shape[0], 0:block.shape[1]/2])
                    subblock4_mean = np.mean(block[block.shape[0]/2:block.shape[0], block.shape[1]/2:block.shape[1]])
                    subblocks_means = np.array([subblock1_mean, subblock2_mean, subblock3_mean, subblock4_mean])

                    m_values = np.abs(subblocks_means.dot(direction_filters))
                    direction_index = m_values.argmax(axis=0)
                    if m_values[direction_index] > direction_threshold:
                        hists[subimage_index, direction_index] += 1

                    #DEBUG
                    # print '\t\tblock means:', subblocks_means
                    # print '\t\tm values:', m_values
                    # print '\t\thistogram', hists[subimage_index, :]
                    # cv2.waitKey(0)

                    jj_begin = jj_end

                ii_begin = ii_end

            subimage_index += 1

    print 'Overall edge histograms'
    print hists
