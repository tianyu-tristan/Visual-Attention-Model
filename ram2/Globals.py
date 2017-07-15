constants = {
    'batchSize': 128,
    'numClasses': 10,
    'learningRate': 0.001,

    # for glimpse network
    'imageSize': 28,
    'imageChannel': 1,
    'largestGlimpseOneSide': 9,
    'smallestGlimpseOneSide': 3,
    'numGlimpseResolution': 3,
    'glimpseOutputSize': 18,

    # for core rnn network
    'numGlimpses': 7,
    'hiddenDim': 256,

    # for location network
    'unitPixels': 12,  # how far from the center is glimpse allowed
    'locationStd': 0.5,
    'alpha': 1.0,
}
