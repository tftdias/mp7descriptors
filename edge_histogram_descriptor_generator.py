from networkx.exception import NetworkXNoPath

__author__ = 'Tiago'


import cv2
import numpy as np
import sys


DIRECTION_FILTERS = np.array([[1, -1, 1, -1],                       #vertical
                              [1, 1, -1, -1],                       #horizontal
                              [np.sqrt(2), 0, 0, -np.sqrt(2)],      #45 diagonal
                              [0, np.sqrt(2), -np.sqrt(2), 0],      #135 diagonal
                              [2, -2, -2, 2]                        #nao direccional
                              ]).T
DIRECTION_THRES = 50

QUANTIZER_MATRIX = np.array([[0.010867, 0.012266, 0.004193, 0.004174, 0.006778],
                             [0.057915, 0.069934, 0.025852, 0.025924, 0.051667],
                             [0.099526, 0.125879, 0.046860, 0.046232, 0.108650],
                             [0.144849, 0.182307, 0.068519, 0.067163, 0.166257],
                             [0.195573, 0.243396, 0.093286, 0.089655, 0.224226],
                             [0.260504, 0.314563, 0.123490, 0.115391, 0.285691],
                             [0.358031, 0.411728, 0.161505, 0.151904, 0.356375],
                             [0.530128, 0.564319, 0.228960, 0.217745, 0.450972]])


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

    hists = np.zeros([16, 5])
    
    subimage_height = image.shape[0]/4
    subimage_width = image.shape[1]/4

    block_size = np.floor(np.sqrt(subimage_height*subimage_width/1100)/2)*2
    block_width = block_size
    block_height = block_size
    blocks_per_subimage = subimage_height/block_height * subimage_width/block_width

    subimage_index = 0
    block_index = 0

    for i in np.arange(0, image.shape[0], subimage_height):
        for j in np.arange(0, image.shape[1], subimage_width):
            subimage = image[i:i+subimage_height, j:j+subimage_width]
            
            #DEBUG
            # cv2.namedWindow('SubImage')
            # print 'Subimage', subimage_index, 'from image - [', i, ':', i+subimage_height, ',', j, ':', j+subimage_width, ']'
            # cv2.imshow('SubImage', subimage)
            # cv2.waitKey(0)

            for ii in np.arange(0, subimage.shape[0], block_height):
                for jj in np.arange(0, subimage.shape[1], block_width):
                    block = subimage[ii:ii+block_height, jj:jj+block_width]

                    #DEBUG
                    # cv2.namedWindow('Block')
                    # print '\tBlock', ii_begin/common_block_height*33+jj_begin/common_block_width, '- size', block.shape[0], 'x', block.shape[1]
                    # cv2.imshow('Block', block)

                    subblock1_mean = np.mean(block[0:block.shape[0]/2, 0:block.shape[1]/2])
                    subblock2_mean = np.mean(block[0:block.shape[0]/2, block.shape[1]/2:block.shape[1]])
                    subblock3_mean = np.mean(block[block.shape[0]/2:block.shape[0], 0:block.shape[1]/2])
                    subblock4_mean = np.mean(block[block.shape[0]/2:block.shape[0], block.shape[1]/2:block.shape[1]])
                    subblocks_means = np.array([subblock1_mean, subblock2_mean, subblock3_mean, subblock4_mean])

                    m_values = np.abs(subblocks_means.dot(DIRECTION_FILTERS))
                    direction_index = m_values.argmax(axis=0)
                    if m_values[direction_index] > DIRECTION_THRES:
                        hists[subimage_index, direction_index] += 1

                    #DEBUG
                    # print '\t\tblock means:', subblocks_means
                    # print '\t\tm values:', m_values
                    # print '\t\thistogram', hists[subimage_index, :]
                    # cv2.waitKey(0)

                    block_index += 1

            subimage_index += 1

    hists = hists / blocks_per_subimage

    quant_hist = np.zeros([16*5], dtype=np.uint8)
    for i in range(16):
        quant_hist[(i*5):(i*5+5)] += np.abs(np.ones([8, 1])*hists[i, :] - QUANTIZER_MATRIX).argmin(axis=0)

    global_hist = np.sum(hists, axis=0) / 16
    quant_global_hist = np.abs(np.ones([8, 1])*global_hist - QUANTIZER_MATRIX).argmin(axis=0)

    semiglobal_hist = np.array([(hists[0, :] + hists[4, :] + hists[8, :] + hists[12, :]) / 4,
                                (hists[1, :] + hists[5, :] + hists[9, :] + hists[13, :]) / 4,
                                (hists[2, :] + hists[6, :] + hists[10, :] + hists[14, :]) / 4,
                                (hists[3, :] + hists[7, :] + hists[9, :] + hists[15, :]) / 4,
                                (hists[0, :] + hists[1, :] + hists[2, :] + hists[3, :]) / 4,
                                (hists[4, :] + hists[5, :] + hists[6, :] + hists[7, :]) / 4,
                                (hists[8, :] + hists[9, :] + hists[10, :] + hists[11, :]) / 4,
                                (hists[12, :] + hists[13, :] + hists[14, :] + hists[15, :]) / 4,
                                (hists[0, :] + hists[1, :] + hists[4, :] + hists[5, :]) / 4,
                                (hists[2, :] + hists[3, :] + hists[6, :] + hists[7, :]) / 4,
                                (hists[8, :] + hists[9, :] + hists[12, :] + hists[13, :]) / 4,
                                (hists[10, :] + hists[11, :] + hists[14, :] + hists[15, :]) / 4,
                                (hists[5, :] + hists[6, :] + hists[9, :] + hists[10, :]) / 4])

    quant_semiglobal_hist = np.zeros([13*5], dtype=np.uint8)
    for i in range(13):
        quant_semiglobal_hist[(i*5):(i*5+5)] += np.abs(np.ones([8, 1])*semiglobal_hist[i, :] - QUANTIZER_MATRIX).argmin(axis=0)

    print 'Local histograms'
    print hists
    print 'Quantified histogram'
    print quant_hist.reshape([16, 5])
    print 'Global histogram'
    print global_hist
    print 'Quantified global histogram'
    print quant_global_hist
    print 'Semiglobal histograms'
    print semiglobal_hist
    print 'Quantified semiglobal histogram'
    print quant_semiglobal_hist
    print 'Number of subimages -', subimage_index
    print 'Number of blocks -', block_index
