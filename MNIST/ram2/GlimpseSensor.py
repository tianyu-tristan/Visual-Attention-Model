import tensorflow as tf
import numpy as np
from Globals import *

class GlimpseSensor(object):

    def __init__(self):
        self.imageSize = constants['imageSize']
        self.batchSize = constants['batchSize']
        self.glimpseOutputSize = constants['glimpseOutputSize']
        self.largestGlimpseOneSide = constants['largestGlimpseOneSide']
        self.smallestGlimpseOneSide = constants['smallestGlimpseOneSide']
        self.numGlimpseResolution = constants['numGlimpseResolution']
        self.glimpseSizes = np.linspace(
            self.smallestGlimpseOneSide,
            self.largestGlimpseOneSide,
            self.numGlimpseResolution).tolist()
        self.imgsWithBbox = []
        self.glimpses = []

    def glimpseAtOneResolution(self, img, centerX, centerY, boxSizeOneSided):
        glimpse = tf.image.crop_to_bounding_box(
            img,
            centerY - boxSizeOneSided,
            centerX - boxSizeOneSided,
            boxSizeOneSided * 2,
            boxSizeOneSided * 2)
        glimpse = tf.expand_dims(glimpse, 0)
        glimpse = tf.image.resize_nearest_neighbor(
            glimpse, (self.glimpseOutputSize, self.glimpseOutputSize))
        return glimpse[0, :, :, :]

    def padImage(self, img, centerX, centerY, imageSize, largestGlimpseOneSide):
        paddingLeft = 0
        paddingRight = 0
        paddingTop = 0
        paddingDown = 0
        if centerX - largestGlimpseOneSide < 0:
            paddingLeft = largestGlimpseOneSide - centerX
        if centerX + largestGlimpseOneSide > imageSize:
            paddingRight = centerX + largestGlimpseOneSide - imageSize
        if centerY - largestGlimpseOneSide < 0:
            paddingTop = largestGlimpseOneSide - centerY
        if centerY + largestGlimpseOneSide > imageSize:
            paddingDown = centerY + largestGlimpseOneSide - imageSize
        # make padding square
        if paddingLeft > paddingTop:
            paddingTop = paddingLeft
        else:
            paddingLeft = paddingTop
        if paddingRight > paddingDown:
            paddingDown = paddingRight
        else:
            paddingRight = paddingDown
        imageWidth = paddingRight + imageSize + paddingLeft
        imageHeight = paddingDown + imageSize + paddingTop
        newX = paddingLeft + centerX
        newY = paddingTop + centerY
        img = tf.image.pad_to_bounding_box(
            img, paddingTop, paddingLeft, imageHeight, imageWidth)
        return img, newX, newY, imageWidth, imageHeight

    # pads the image, return the new location coordinates after padding
    # collects all images with glimpse bounding box in a list
    # returns the padded image
    def prepareImage(self, img, X, Y):
        X = self.imageSize * float(X - (-1.0)) / 2.0
        Y = self.imageSize * float(Y - (-1.0)) / 2.0
        img, newX, newY, imageWidth, imageHeight = self.padImage(
            img, int(X), int(Y), self.imageSize, self.largestGlimpseOneSide)
        # draw glimpse bounding boxes
        bboxes = []
        for boxSizeOneSided in self.glimpseSizes:
            bbox = np.asarray(
                [
                    float(
                        newY -
                        boxSizeOneSided) /
                    float(imageHeight),
                    float(
                        newX -
                        boxSizeOneSided) /
                    float(imageWidth),
                    float(
                        newY +
                        boxSizeOneSided) /
                    float(imageHeight),
                    float(
                        newX +
                        boxSizeOneSided) /
                    float(imageWidth)])
            bboxes.append(bbox)
        bboxes = np.vstack(bboxes)
        bboxes = tf.to_float(bboxes)
        bboxes = tf.expand_dims(bboxes, 0)
        img = tf.image.convert_image_dtype(img, tf.float32)
        imgWtihBbox = tf.expand_dims(img, 0)
        imgWtihBbox = tf.image.draw_bounding_boxes(imgWtihBbox, bboxes)
        imgWtihBbox = tf.image.resize_nearest_neighbor(imgWtihBbox, (self.imageSize, self.imageSize))
        imgWtihBbox = imgWtihBbox[0, :, :, :]
        self.imgsWithBbox.append(imgWtihBbox)
        # focus location change due to padding
        return img, newX, newY

    # extract the glimpses and collect them in a list
    def extractGlimpses(self, img, newX, newY):
        # take a glimpse (has different resolutions)
        glimpseAllResolution = []
        for boxSizeOneSided in self.glimpseSizes:
            g = self.glimpseAtOneResolution(
                img, newX, newY, int(boxSizeOneSided))
            glimpseAllResolution.append(g)
        g = tf.concat(glimpseAllResolution, axis=2)
        self.glimpses.append(g)

    def extract(self, imgs, imgsPlaceholder, locations):
        locations = tf.get_default_session().run(locations, feed_dict={imgsPlaceholder: imgs})
        imgs = tf.reshape(imgsPlaceholder, [self.batchSize, self.imageSize, self.imageSize, 1])
        imgs = tf.unstack(imgs)
        self.imgsWithBbox[:] = []
        self.glimpses[:] = []
        assert len(imgs) == len(locations)
        for i in range(len(imgs)):
            locationX = locations[i][0]
            locationY = locations[i][1]
            assert locationX > -1
            assert locationX < 1
            assert locationY > -1
            assert locationY < 1
            img, newX, newY = self.prepareImage(
                imgs[i], locationX, locationY)
            self.extractGlimpses(img, newX, newY)
        extractedGlimpses = tf.unstack(self.glimpses)
        tensorboardLabel = 'glimpse'
        tf.summary.image(
            tensorboardLabel + '/bbox',
            tf.unstack(
                self.imgsWithBbox),
                max_outputs=10)
        tf.summary.image(
            tensorboardLabel + '/cropped',
            extractedGlimpses,
            max_outputs=10)
        return extractedGlimpses
