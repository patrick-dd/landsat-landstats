import cPickle
import numpy as np

batch_size = 500

test_y = cPickle.load(file('y_interpolated_20.save', 'rb'))
test_loc = cPickle.load(file('location_20.save', 'rb'))


print min(test_y)
print max(test_y)
